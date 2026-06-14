"""Reusable Bayesian optimization primitives.

This module is domain-neutral. It knows how to sample a search space, evaluate
an objective with optional repeats, fit a noisy Gaussian Process surrogate, and
select candidates with Expected Improvement. Domain modules provide metrics,
constraints, reports, and optional typed observation subclasses.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


Config = dict[str, float | str]
Objective = Callable[[Config], tuple[float, dict[str, Any]]]


@dataclass(frozen=True)
class Parameter:
    """A tunable parameter in a search space."""

    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()

    def sample(self, rng: random.Random) -> float | str:
        if self.kind == "float":
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name} float parameter needs low/high bounds.")
            return rng.uniform(self.low, self.high)
        if self.kind == "int":
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name} int parameter needs low/high bounds.")
            return rng.randint(int(self.low), int(self.high))
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"{self.name} categorical parameter needs choices.")
            return rng.choice(self.choices)
        raise ValueError(f"Unsupported parameter kind: {self.kind}")

    def encode(self, value: float | str) -> list[float]:
        if self.kind == "categorical":
            return [1.0 if value == choice else 0.0 for choice in self.choices]
        numeric_value = float(value)
        if self.low is None or self.high is None or self.high == self.low:
            return [numeric_value]
        return [(numeric_value - self.low) / (self.high - self.low)]


@dataclass(frozen=True)
class SearchSpace:
    """A collection of tunable parameters."""

    parameters: tuple[Parameter, ...]

    def sample(self, rng: random.Random) -> Config:
        return {parameter.name: parameter.sample(rng) for parameter in self.parameters}

    def encode(self, config: Config) -> list[float]:
        vector: list[float] = []
        for parameter in self.parameters:
            vector.extend(parameter.encode(config[parameter.name]))
        return vector


@dataclass(frozen=True)
class Constraint:
    """A domain metric constraint used by constrained Bayesian optimization.

    The optimizer models each constraint as a slack value where positive means
    feasible. For example, ``kind="min"`` with ``threshold=0.8`` means
    ``metadata[metric] >= 0.8``.
    """

    metric: str
    kind: str
    threshold: float

    def slack(self, metadata: dict[str, Any]) -> float:
        value = float(metadata[self.metric])
        if self.kind == "min":
            return value - self.threshold
        if self.kind == "max":
            return self.threshold - value
        raise ValueError(f"Unsupported constraint kind: {self.kind}")

    def is_feasible(self, metadata: dict[str, Any]) -> bool:
        return self.slack(metadata) >= 0.0


@dataclass(frozen=True)
class BayesianOptimizationConfig:
    """Domain-neutral Bayesian optimization policy."""

    iterations: int = 35
    initial_points: int = 8
    candidate_pool: int = 500
    observation_repeats: int = 1
    gp_noise: float = 1e-4
    exploration: float = 0.01
    use_constrained_acquisition: bool = True
    random_state: int = 42
    evaluate_baseline: bool = True


@dataclass(frozen=True)
class GenericObservation:
    """One evaluated configuration with domain metrics in metadata."""

    iteration: int
    config: Config
    objective: float
    objective_std: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GenericOptimizationResult:
    """Generic optimization result for non-domain-specific use."""

    best: GenericObservation
    observations: list[GenericObservation]
    baseline: GenericObservation | None = None


@dataclass
class BayesianOptimizer:
    """Reusable Bayesian optimizer.

    Subclass this for domain-specific observation shaping or feasibility rules.
    """

    search_space: SearchSpace
    objective: Objective
    config: BayesianOptimizationConfig = BayesianOptimizationConfig()
    baseline_config: Config | None = None
    constraints: tuple[Constraint, ...] = ()

    def run(
        self,
        initial_observations: Sequence[GenericObservation] | None = None,
        baseline: GenericObservation | None = None,
    ) -> GenericOptimizationResult:
        rng = random.Random(self.config.random_state)
        observations = list(initial_observations or [])
        if baseline is None and self.config.evaluate_baseline and self.baseline_config is not None:
            baseline = self._evaluate(-1, self.baseline_config)
        next_iteration = max((observation.iteration for observation in observations), default=-1) + 1
        new_random_points = max(0, self.config.initial_points - len(observations))
        for offset in range(self.config.iterations):
            if offset < new_random_points or len(observations) < 3:
                candidate = self.search_space.sample(rng)
            else:
                candidate = self._suggest(observations, rng)
            observations.append(self._evaluate(next_iteration + offset, candidate))
        feasible = [observation for observation in observations if self._is_feasible(observation)]
        best_pool = feasible or observations
        return GenericOptimizationResult(
            best=min(best_pool, key=lambda item: item.objective),
            observations=observations,
            baseline=baseline,
        )

    def _is_feasible(self, observation: GenericObservation) -> bool:
        return all(constraint.is_feasible(observation.metadata) for constraint in self.constraints)

    def _make_observation(
        self,
        iteration: int,
        candidate: Config,
        objective: float,
        metadata: dict[str, Any],
    ) -> GenericObservation:
        return GenericObservation(
            iteration=iteration,
            config=candidate,
            objective=objective,
            objective_std=float(metadata.get("objective_std", 0.0)),
            metadata=metadata,
        )

    def _evaluate(self, iteration: int, candidate: Config) -> GenericObservation:
        objective, metadata = self._evaluate_repeated(candidate)
        return self._make_observation(iteration, candidate, objective, metadata)

    def _evaluate_repeated(self, candidate: Config) -> tuple[float, dict[str, Any]]:
        repeats = max(1, self.config.observation_repeats)
        evaluations = [self.objective(candidate) for _ in range(repeats)]
        if repeats == 1:
            objective, metadata = evaluations[0]
            metadata = {**metadata, "objective_std": float(metadata.get("objective_std", 0.0)), "repeats": 1}
            return objective, metadata

        objectives = [value for value, _ in evaluations]
        metadata = aggregate_metadata([item for _, item in evaluations])
        metadata["repeats"] = repeats
        metadata["objective_std"] = sample_std(objectives)
        metadata["replicate_objectives"] = objectives
        return mean(objectives), metadata

    def _suggest(self, observations: Sequence[GenericObservation], rng: random.Random) -> Config:
        x_train = [self.search_space.encode(observation.config) for observation in observations]
        y_train = [observation.objective for observation in observations]
        objective_noise = [
            max(self.config.gp_noise, observation.objective_std)
            / math.sqrt(max(1, observation.metadata.get("repeats", 1)))
            for observation in observations
        ]
        model = GaussianProcessRegressor.fit(
            x_train,
            y_train,
            objective_noise=objective_noise,
            base_noise=self.config.gp_noise,
        )
        best = min(model.predict(x_train[index])[0] for index in range(len(x_train)))
        constraint_models = self._fit_constraint_models(observations, x_train)
        candidates = [self.search_space.sample(rng) for _ in range(self.config.candidate_pool)]
        return max(
            candidates,
            key=lambda config: self._acquisition_score(config, model, best, constraint_models),
        )

    def _fit_constraint_models(
        self,
        observations: Sequence[GenericObservation],
        x_train: list[list[float]],
    ) -> list[tuple[Constraint, GaussianProcessRegressor]]:
        if not self.config.use_constrained_acquisition or not self.constraints:
            return []
        models: list[tuple[Constraint, GaussianProcessRegressor]] = []
        for constraint in self.constraints:
            if not all(constraint.metric in observation.metadata for observation in observations):
                continue
            slack_values = [constraint.slack(observation.metadata) for observation in observations]
            slack_noise = [
                float(observation.metadata.get(f"{constraint.metric}_std", 0.0))
                / math.sqrt(max(1, observation.metadata.get("repeats", 1)))
                for observation in observations
            ]
            models.append(
                (
                    constraint,
                    GaussianProcessRegressor.fit(
                        x_train,
                        slack_values,
                        objective_noise=slack_noise,
                        base_noise=self.config.gp_noise,
                    ),
                )
            )
        return models

    def _acquisition_score(
        self,
        config: Config,
        objective_model: GaussianProcessRegressor,
        best: float,
        constraint_models: Sequence[tuple[Constraint, GaussianProcessRegressor]],
    ) -> float:
        encoded = self.search_space.encode(config)
        improvement = expected_improvement(
            objective_model.predict(encoded),
            best,
            exploration=self.config.exploration,
        )
        if not constraint_models:
            return improvement
        feasibility_probability = 1.0
        for _, model in constraint_models:
            feasibility_probability *= probability_slack_is_feasible(model.predict(encoded))
        return improvement * feasibility_probability


def aggregate_metadata(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric metadata fields and keep sample std for repeated evaluations."""

    if not items:
        return {}
    result: dict[str, Any] = {}
    keys = set().union(*(item.keys() for item in items))
    for key in sorted(keys):
        values = [item[key] for item in items if key in item]
        numeric_values = [float(value) for value in values if isinstance(value, int | float)]
        if len(numeric_values) == len(values) and numeric_values:
            result[key] = mean(numeric_values)
            if len(numeric_values) > 1:
                result[f"{key}_std"] = sample_std(numeric_values)
        elif values:
            result[key] = values[-1]
    return result


@dataclass(frozen=True)
class GaussianProcessRegressor:
    """Tiny GP regressor with RBF kernel, normalized targets, and observation noise."""

    x_train: list[list[float]]
    y_mean: float
    y_std: float
    alpha: list[float]
    inverse_kernel: list[list[float]]
    length_scale: float = 0.55
    noise: float = 1e-4

    @classmethod
    def fit(
        cls,
        x_train: list[list[float]],
        y_train: list[float],
        objective_noise: Sequence[float] | None = None,
        base_noise: float = 1e-4,
        length_scales: Sequence[float] = (0.25, 0.4, 0.65, 1.0, 1.6),
    ) -> GaussianProcessRegressor:
        y_mean = sum(y_train) / len(y_train)
        variance = sum((value - y_mean) ** 2 for value in y_train) / len(y_train)
        y_std = math.sqrt(variance) or 1.0
        y_scaled = [(value - y_mean) / y_std for value in y_train]
        scaled_noise = scaled_noise_variances(objective_noise, y_std, base_noise, len(y_train))
        best_model: GaussianProcessRegressor | None = None
        best_score = float("inf")
        for length_scale in length_scales:
            kernel = [
                [
                    rbf_kernel(left, right, length_scale=length_scale)
                    + (scaled_noise[row] if row == column else 0.0)
                    for column, right in enumerate(x_train)
                ]
                for row, left in enumerate(x_train)
            ]
            inverse_kernel, log_determinant = invert_matrix_with_logdet(kernel)
            alpha = mat_vec(inverse_kernel, y_scaled)
            score = 0.5 * dot(y_scaled, alpha) + 0.5 * log_determinant
            if score < best_score:
                best_score = score
                best_model = cls(
                    x_train=x_train,
                    y_mean=y_mean,
                    y_std=y_std,
                    alpha=alpha,
                    inverse_kernel=inverse_kernel,
                    length_scale=length_scale,
                    noise=base_noise,
                )
        if best_model is None:
            raise ValueError("Could not fit Gaussian process.")
        return best_model

    def predict(self, x: list[float]) -> tuple[float, float]:
        kernel_vector = [rbf_kernel(x, known, length_scale=self.length_scale) for known in self.x_train]
        mean_scaled = dot(kernel_vector, self.alpha)
        variance_scaled = max(1e-9, 1.0 - dot(kernel_vector, mat_vec(self.inverse_kernel, kernel_vector)))
        return self.y_mean + mean_scaled * self.y_std, math.sqrt(variance_scaled) * self.y_std


def scaled_noise_variances(
    objective_noise: Sequence[float] | None,
    y_std: float,
    base_noise: float,
    size: int,
) -> list[float]:
    if objective_noise is None:
        return [base_noise for _ in range(size)]
    return [max(base_noise, (float(noise) / y_std) ** 2) for noise in objective_noise]


def expected_improvement(prediction: tuple[float, float], best: float, exploration: float = 0.0) -> float:
    mean_value, std = prediction
    if std <= 1e-9:
        return 0.0
    improvement = best - mean_value - exploration
    z = improvement / std
    return improvement * normal_cdf(z) + std * normal_pdf(z)


def probability_slack_is_feasible(prediction: tuple[float, float]) -> float:
    mean_value, std = prediction
    if std <= 1e-9:
        return 1.0 if mean_value >= 0.0 else 0.0
    return normal_cdf(mean_value / std)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rbf_kernel(left: Sequence[float], right: Sequence[float], length_scale: float) -> float:
    distance = sum((a - b) ** 2 for a, b in zip(left, right))
    return math.exp(-0.5 * distance / (length_scale**2))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def invert_matrix_with_logdet(matrix: list[list[float]]) -> tuple[list[list[float]], float]:
    size = len(matrix)
    augmented = [
        [float(matrix[row][column]) for column in range(size)]
        + [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]
    log_determinant = 0.0
    determinant_sign = 1.0
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Matrix is singular.")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            determinant_sign *= -1.0
        divisor = augmented[column][column]
        determinant_sign *= 1.0 if divisor > 0 else -1.0
        log_determinant += math.log(abs(divisor))
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    if determinant_sign <= 0:
        log_determinant = float("inf")
    return [row[size:] for row in augmented], log_determinant


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    inverse, _ = invert_matrix_with_logdet(matrix)
    return inverse


def mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
