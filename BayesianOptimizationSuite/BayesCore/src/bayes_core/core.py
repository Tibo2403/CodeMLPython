"""Generic Bayesian optimization primitives.

The core is deliberately domain-agnostic. Projects provide:

1. a SearchSpace;
2. an objective function returning (objective, metadata);
3. optional domain-specific reports built on top of the observations.
"""

from __future__ import annotations

import json
import math
import random
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Config = dict[str, float | str]
Objective = Callable[[Config], tuple[float, dict[str, Any]]]


@dataclass(frozen=True)
class Constraint:
    """A lightweight feasibility rule evaluated from observation metadata."""

    metric: str
    operator: str
    threshold: float

    def is_satisfied(self, metadata: dict[str, Any]) -> bool:
        value = float(metadata[self.metric])
        if self.operator == "min":
            return value >= self.threshold
        if self.operator == "max":
            return value <= self.threshold
        raise ValueError("Constraint operator must be 'min' or 'max'.")


@dataclass(frozen=True)
class Parameter:
    """A tunable parameter."""

    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()

    def sample(self, rng: random.Random) -> float | str:
        if self.kind == "float":
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name} needs low/high bounds.")
            return rng.uniform(self.low, self.high)
        if self.kind == "int":
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name} needs low/high bounds.")
            return rng.randint(int(self.low), int(self.high))
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"{self.name} needs choices.")
            return rng.choice(self.choices)
        raise ValueError(f"Unsupported parameter kind: {self.kind}")

    def encode(self, value: float | str) -> list[float]:
        if self.kind == "categorical":
            return [1.0 if value == choice else 0.0 for choice in self.choices]
        numeric = float(value)
        if self.low is None or self.high is None or self.low == self.high:
            return [numeric]
        return [(numeric - self.low) / (self.high - self.low)]


@dataclass(frozen=True)
class SearchSpace:
    """A collection of optimization parameters."""

    parameters: tuple[Parameter, ...]

    def sample(self, rng: random.Random) -> Config:
        return {parameter.name: parameter.sample(rng) for parameter in self.parameters}

    def encode(self, config: Config) -> list[float]:
        vector: list[float] = []
        for parameter in self.parameters:
            vector.extend(parameter.encode(config[parameter.name]))
        return vector


@dataclass(frozen=True)
class OptimizationConfig:
    """Bayesian optimization policy."""

    iterations: int = 30
    initial_points: int = 8
    candidate_pool: int = 300
    random_state: int = 42
    acquisition: str = "expected_improvement"
    exploration_weight: float = 1.25
    min_candidate_distance: float = 0.04
    observation_repeats: int = 1
    gp_noise: float = 1e-6


@dataclass(frozen=True)
class GenericObservation:
    """One evaluated configuration."""

    iteration: int
    config: Config
    objective: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GenericOptimizationResult:
    """Generic optimization result."""

    best: GenericObservation
    observations: list[GenericObservation]
    optimization_config: OptimizationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": observation_to_dict(self.best),
            "observations": [observation_to_dict(observation) for observation in self.observations],
            "optimization_config": config_to_dict(self.optimization_config),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        metadata_keys = sorted({key for observation in self.observations for key in observation.metadata})
        fieldnames = ["iteration", "objective", "config_json", *metadata_keys]
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for observation in self.observations:
                row = {
                    "iteration": observation.iteration,
                    "objective": observation.objective,
                    "config_json": json.dumps(observation.config, sort_keys=True),
                }
                row.update({key: observation.metadata.get(key, "") for key in metadata_keys})
                writer.writerow(row)

    def pareto_front(self, objectives: dict[str, str]) -> list[GenericObservation]:
        """Return non-dominated observations for metadata metrics.

        `objectives` maps metric names to either `min` or `max`.
        """

        front: list[GenericObservation] = []
        for candidate in self.observations:
            if not any(dominates(other, candidate, objectives) for other in self.observations):
                front.append(candidate)
        return front

    @classmethod
    def from_json(cls, text: str) -> GenericOptimizationResult:
        payload = json.loads(text)
        observations = [observation_from_dict(item) for item in payload["observations"]]
        return cls(
            best=observation_from_dict(payload["best"]),
            observations=observations,
            optimization_config=optimization_config_from_dict(payload["optimization_config"]),
        )


@dataclass
class GenericBayesianOptimizer:
    """Reusable Bayesian optimizer."""

    search_space: SearchSpace
    objective: Objective
    config: OptimizationConfig = OptimizationConfig()
    constraints: tuple[Constraint, ...] = ()

    def run(self, initial_observations: Sequence[GenericObservation] = ()) -> GenericOptimizationResult:
        rng = random.Random(self.config.random_state)
        observations = list(initial_observations)
        start_iteration = len(observations)
        for iteration in range(start_iteration, start_iteration + self.config.iterations):
            if iteration < self.config.initial_points or len(observations) < 3:
                candidate = self.search_space.sample(rng)
            else:
                candidate = self._suggest(observations, rng)
            observations.append(self._evaluate(iteration, candidate))
        return GenericOptimizationResult(
            best=self._best_observation(observations),
            observations=observations,
            optimization_config=self.config,
        )

    def run_random_baseline(self) -> GenericOptimizationResult:
        baseline_config = OptimizationConfig(
            iterations=self.config.iterations,
            initial_points=self.config.iterations,
            candidate_pool=self.config.candidate_pool,
            random_state=self.config.random_state,
            acquisition="random_search",
            exploration_weight=self.config.exploration_weight,
            min_candidate_distance=self.config.min_candidate_distance,
            observation_repeats=self.config.observation_repeats,
            gp_noise=self.config.gp_noise,
        )
        return GenericBayesianOptimizer(
            search_space=self.search_space,
            objective=self.objective,
            config=baseline_config,
            constraints=self.constraints,
        ).run()

    def _evaluate(self, iteration: int, config: Config) -> GenericObservation:
        objective, metadata = self._evaluate_repeated(config)
        return GenericObservation(
            iteration=iteration,
            config=config,
            objective=float(objective),
            metadata=metadata,
        )

    def _evaluate_repeated(self, config: Config) -> tuple[float, dict[str, Any]]:
        repeats = max(1, self.config.observation_repeats)
        evaluations = [self.objective(config) for _ in range(repeats)]
        if repeats == 1:
            objective, metadata = evaluations[0]
            return float(objective), {
                **metadata,
                "objective_std": float(metadata.get("objective_std", 0.0)),
                "repeats": 1,
            }
        objectives = [float(objective) for objective, _ in evaluations]
        metadata = aggregate_metadata([metadata for _, metadata in evaluations])
        metadata["objective_std"] = sample_std(objectives)
        metadata["replicate_objectives"] = objectives
        metadata["repeats"] = repeats
        return mean(objectives), metadata

    def _suggest(self, observations: Sequence[GenericObservation], rng: random.Random) -> Config:
        if self.config.acquisition == "random_search":
            return self.search_space.sample(rng)
        x_train = [self.search_space.encode(observation.config) for observation in observations]
        y_train = [observation.objective for observation in observations]
        objective_noise = [
            max(self.config.gp_noise, float(observation.metadata.get("objective_std", 0.0)))
            / math.sqrt(max(1, int(observation.metadata.get("repeats", 1))))
            for observation in observations
        ]
        model = GaussianProcessRegressor.fit(
            x_train,
            y_train,
            objective_noise=objective_noise,
            base_noise=self.config.gp_noise,
        )
        best = min(y_train)
        candidates = [self.search_space.sample(rng) for _ in range(self.config.candidate_pool)]
        return max(
            candidates,
            key=lambda candidate: self._acquisition_score(candidate, model, best, x_train),
        )

    def _best_observation(self, observations: Sequence[GenericObservation]) -> GenericObservation:
        feasible = [observation for observation in observations if self._is_feasible(observation)]
        candidates = feasible or list(observations)
        return min(candidates, key=lambda item: item.objective)

    def _is_feasible(self, observation: GenericObservation) -> bool:
        return all(constraint.is_satisfied(observation.metadata) for constraint in self.constraints)

    def _acquisition_score(
        self,
        config: Config,
        model: GaussianProcessRegressor,
        best: float,
        x_train: Sequence[Sequence[float]],
    ) -> float:
        vector = self.search_space.encode(config)
        score = acquisition_score(
            model.predict(vector),
            best=best,
            strategy=self.config.acquisition,
            exploration_weight=self.config.exploration_weight,
        )
        distance = min(euclidean_distance(vector, known) for known in x_train)
        if distance < self.config.min_candidate_distance:
            score *= distance / max(self.config.min_candidate_distance, 1e-9)
        return score


@dataclass(frozen=True)
class GaussianProcessRegressor:
    """Tiny Gaussian Process regressor with an RBF kernel."""

    x_train: list[list[float]]
    y_mean: float
    y_std: float
    alpha: list[float]
    inverse_kernel: list[list[float]]
    length_scale: float = 0.55
    noise: float = 1e-6

    @classmethod
    def fit(
        cls,
        x_train: list[list[float]],
        y_train: list[float],
        objective_noise: Sequence[float] | None = None,
        base_noise: float = 1e-6,
    ) -> GaussianProcessRegressor:
        y_mean = sum(y_train) / len(y_train)
        variance = sum((value - y_mean) ** 2 for value in y_train) / len(y_train)
        y_std = math.sqrt(variance) or 1.0
        y_scaled = [(value - y_mean) / y_std for value in y_train]
        scaled_noise = scaled_noise_variances(objective_noise, y_std, base_noise, len(y_train))
        kernel = [
            [
                rbf_kernel(left, right, length_scale=0.55) + (scaled_noise[row] if row == column else 0.0)
                for column, right in enumerate(x_train)
            ]
            for row, left in enumerate(x_train)
        ]
        inverse_kernel = invert_matrix(kernel)
        return cls(
            x_train=x_train,
            y_mean=y_mean,
            y_std=y_std,
            alpha=mat_vec(inverse_kernel, y_scaled),
            inverse_kernel=inverse_kernel,
            noise=base_noise,
        )

    def predict(self, x: list[float]) -> tuple[float, float]:
        kernel_vector = [rbf_kernel(x, known, length_scale=self.length_scale) for known in self.x_train]
        mean_scaled = dot(kernel_vector, self.alpha)
        variance_scaled = max(1e-9, 1.0 - dot(kernel_vector, mat_vec(self.inverse_kernel, kernel_vector)))
        return self.y_mean + mean_scaled * self.y_std, math.sqrt(variance_scaled) * self.y_std


def expected_improvement(prediction: tuple[float, float], best: float) -> float:
    mean, std = prediction
    if std <= 1e-9:
        return 0.0
    improvement = best - mean
    z = improvement / std
    return improvement * normal_cdf(z) + std * normal_pdf(z)


def probability_improvement(prediction: tuple[float, float], best: float) -> float:
    mean, std = prediction
    if std <= 1e-9:
        return 0.0
    return normal_cdf((best - mean) / std)


def lower_confidence_bound(prediction: tuple[float, float], exploration_weight: float = 1.25) -> float:
    mean, std = prediction
    return -(mean - exploration_weight * std)


def acquisition_score(
    prediction: tuple[float, float],
    best: float,
    strategy: str = "expected_improvement",
    exploration_weight: float = 1.25,
) -> float:
    if strategy == "expected_improvement":
        return expected_improvement(prediction, best)
    if strategy == "probability_improvement":
        return probability_improvement(prediction, best)
    if strategy == "lower_confidence_bound":
        return lower_confidence_bound(prediction, exploration_weight)
    if strategy == "random_search":
        return 0.0
    raise ValueError("Unknown acquisition strategy.")


def rbf_kernel(left: Sequence[float], right: Sequence[float], length_scale: float) -> float:
    distance = sum((a - b) ** 2 for a, b in zip(left, right))
    return math.exp(-0.5 * distance / (length_scale**2))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [
        [float(matrix[row][column]) for column in range(size)]
        + [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Matrix is singular.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [row[size:] for row in augmented]


def mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def observation_to_dict(observation: GenericObservation) -> dict[str, Any]:
    return {
        "iteration": observation.iteration,
        "config": observation.config,
        "objective": observation.objective,
        "metadata": observation.metadata,
    }


def observation_from_dict(payload: dict[str, Any]) -> GenericObservation:
    return GenericObservation(
        iteration=int(payload["iteration"]),
        config=dict(payload["config"]),
        objective=float(payload["objective"]),
        metadata=dict(payload["metadata"]),
    )


def config_to_dict(config: OptimizationConfig) -> dict[str, Any]:
    return {
        "iterations": config.iterations,
        "initial_points": config.initial_points,
        "candidate_pool": config.candidate_pool,
        "random_state": config.random_state,
        "acquisition": config.acquisition,
        "exploration_weight": config.exploration_weight,
        "min_candidate_distance": config.min_candidate_distance,
        "observation_repeats": config.observation_repeats,
        "gp_noise": config.gp_noise,
    }


def optimization_config_from_dict(payload: dict[str, Any]) -> OptimizationConfig:
    return OptimizationConfig(
        iterations=int(payload.get("iterations", 30)),
        initial_points=int(payload.get("initial_points", 8)),
        candidate_pool=int(payload.get("candidate_pool", 300)),
        random_state=int(payload.get("random_state", 42)),
        acquisition=str(payload.get("acquisition", "expected_improvement")),
        exploration_weight=float(payload.get("exploration_weight", 1.25)),
        min_candidate_distance=float(payload.get("min_candidate_distance", 0.04)),
        observation_repeats=int(payload.get("observation_repeats", 1)),
        gp_noise=float(payload.get("gp_noise", 1e-6)),
    )


def dominates(
    left: GenericObservation,
    right: GenericObservation,
    objectives: dict[str, str],
) -> bool:
    better_or_equal = True
    strictly_better = False
    for metric, direction in objectives.items():
        left_value = float(left.metadata[metric])
        right_value = float(right.metadata[metric])
        if direction == "min":
            better_or_equal = better_or_equal and left_value <= right_value
            strictly_better = strictly_better or left_value < right_value
        elif direction == "max":
            better_or_equal = better_or_equal and left_value >= right_value
            strictly_better = strictly_better or left_value > right_value
        else:
            raise ValueError("Pareto objective direction must be 'min' or 'max'.")
    return better_or_equal and strictly_better


def aggregate_metadata(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric metadata and keep the last value for non-numeric fields."""

    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    aggregated: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items if key in item]
        if all(isinstance(value, int | float) for value in values):
            numeric_values = [float(value) for value in values]
            aggregated[key] = mean(numeric_values)
            if len(numeric_values) > 1:
                aggregated[f"{key}_std"] = sample_std(numeric_values)
        else:
            aggregated[key] = values[-1]
    return aggregated


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def scaled_noise_variances(
    objective_noise: Sequence[float] | None,
    y_std: float,
    base_noise: float,
    size: int,
) -> list[float]:
    if objective_noise is None:
        return [base_noise for _ in range(size)]
    return [max(base_noise, (float(noise) / y_std) ** 2) for noise in objective_noise]
