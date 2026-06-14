"""Bayesian optimization engine for token-budget tuning.

The optimizer searches over prompt/context settings and minimizes a penalized
cost: token usage plus a quality-violation penalty. It uses a small Gaussian
process surrogate with an RBF kernel and Expected Improvement acquisition.
The implementation is dependency-light so the project remains easy to run.
"""

from __future__ import annotations

import csv
import html
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Config = dict[str, float | str]
Objective = Callable[[Config], tuple[float, dict[str, Any]]]


@dataclass(frozen=True)
class Task:
    """A representative AI workload used to evaluate a configuration."""

    name: str
    kind: str
    prompt: str
    quality_weight: float = 1.0
    token_multiplier: float = 1.0
    requires_retrieval: bool = False
    requires_examples: bool = False
    requires_reasoning: bool = False


@dataclass(frozen=True)
class Parameter:
    """A tunable parameter in the token optimization space."""

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

    @classmethod
    def token_prompt_space(cls) -> SearchSpace:
        """Default search space for prompt/context compression."""

        return cls(
            parameters=(
                Parameter("max_context_chars", "int", low=800, high=8000),
                Parameter("retrieval_top_k", "int", low=1, high=10),
                Parameter("summary_ratio", "float", low=0.15, high=1.0),
                Parameter("few_shot_examples", "int", low=0, high=5),
                Parameter("reasoning_level", "categorical", choices=("minimal", "low", "medium")),
                Parameter("format_style", "categorical", choices=("compact", "balanced", "verbose")),
            )
        )


def default_baseline_config() -> Config:
    """Return a conservative baseline prompt configuration."""

    return {
        "max_context_chars": 8000,
        "retrieval_top_k": 8,
        "summary_ratio": 1.0,
        "few_shot_examples": 3,
        "reasoning_level": "medium",
        "format_style": "balanced",
    }


@dataclass(frozen=True)
class Observation:
    """One evaluated configuration."""

    iteration: int
    config: Config
    objective: float
    tokens: int
    quality: float
    latency_ms: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OptimizationResult:
    """Final optimization output."""

    best: Observation
    observations: list[Observation]
    baseline: Observation | None = None
    optimization_config: TokenOptimizationConfig | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "best": _observation_to_dict(self.best),
                "baseline": _observation_to_dict(self.baseline) if self.baseline else None,
                "optimization_config": _config_to_dict(self.optimization_config)
                if self.optimization_config
                else None,
                "savings": self.savings_summary(),
                "observations": [_observation_to_dict(observation) for observation in self.observations],
            },
            indent=2,
            sort_keys=True,
        )

    def savings_summary(self) -> dict[str, float | int | None]:
        if self.baseline is None:
            return {
                "tokens_saved": None,
                "token_reduction_percent": None,
                "latency_ms_saved": None,
                "quality_delta": None,
            }
        tokens_saved = self.baseline.tokens - self.best.tokens
        latency_saved = self.baseline.latency_ms - self.best.latency_ms
        return {
            "tokens_saved": tokens_saved,
            "token_reduction_percent": round(tokens_saved / self.baseline.tokens * 100.0, 2)
            if self.baseline.tokens
            else None,
            "latency_ms_saved": round(latency_saved, 2),
            "quality_delta": round(self.best.quality - self.baseline.quality, 4),
        }

    def pareto_front(self) -> list[Observation]:
        """Return non-dominated observations for token/quality trade-off."""

        front: list[Observation] = []
        for candidate in self.observations:
            dominated = any(
                other.tokens <= candidate.tokens
                and other.quality >= candidate.quality
                and (other.tokens < candidate.tokens or other.quality > candidate.quality)
                for other in self.observations
            )
            if not dominated:
                front.append(candidate)
        return sorted(front, key=lambda item: item.tokens)

    def write_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "iteration",
                    "objective",
                    "tokens",
                    "quality",
                    "latency_ms",
                    "config_json",
                    "metadata_json",
                ],
            )
            writer.writeheader()
            for observation in self.observations:
                writer.writerow(
                    {
                        "iteration": observation.iteration,
                        "objective": f"{observation.objective:.6f}",
                        "tokens": observation.tokens,
                        "quality": f"{observation.quality:.6f}",
                        "latency_ms": f"{observation.latency_ms:.2f}",
                        "config_json": json.dumps(observation.config, sort_keys=True),
                        "metadata_json": json.dumps(observation.metadata, sort_keys=True),
                    }
                )

    def write_markdown_report(self, path: Path) -> None:
        feasible_count = sum(
            1
            for observation in self.observations
            if observation.quality >= float(observation.metadata.get("quality_floor", 0.0))
        )
        lines = [
            "# Token Optimization Report",
            "",
            "## Savings",
            "",
        ]
        savings = self.savings_summary()
        if self.baseline is not None:
            lines.extend(
                [
                    f"- Baseline tokens: {self.baseline.tokens}",
                    f"- Optimized tokens: {self.best.tokens}",
                    f"- Tokens saved: {savings['tokens_saved']}",
                    f"- Token reduction: {savings['token_reduction_percent']}%",
                    f"- Latency saved: {savings['latency_ms_saved']} ms",
                    f"- Quality delta: {savings['quality_delta']}",
                    "",
                ]
            )
        else:
            lines.extend(["- No baseline was evaluated.", ""])
        lines.extend(
            [
            "## Best Configuration",
            "",
            f"- Tokens: {self.best.tokens}",
            f"- Quality: {self.best.quality:.4f}",
            f"- Objective: {self.best.objective:.4f}",
            f"- Latency estimate: {self.best.latency_ms:.2f} ms",
            f"- Feasible evaluations: {feasible_count}/{len(self.observations)}",
            "",
            "```json",
            json.dumps(self.best.config, indent=2, sort_keys=True),
            "```",
            "",
            "## Best Task Breakdown",
            "",
        ]
        )
        if self.best.metadata.get("tasks"):
            lines.extend(
                [
                    "| Task | Kind | Tokens | Quality |",
                    "| --- | --- | ---: | ---: |",
                ]
            )
            for task in self.best.metadata["tasks"]:
                lines.append(
                    f"| {task['name']} | {task['kind']} | {task['tokens']} | {float(task['quality']):.4f} |"
                )
        else:
            lines.append("No task-level breakdown available.")
        lines.extend(
            [
            "",
            "## Baseline Configuration",
            "",
            "```json",
            json.dumps(self.baseline.config if self.baseline else {}, indent=2, sort_keys=True),
            "```",
            "",
            "## Top Evaluations",
            "",
            "| Iteration | Tokens | Quality | Objective |",
            "| --- | ---: | ---: | ---: |",
            ]
        )
        for observation in sorted(self.observations, key=lambda item: item.objective)[:10]:
            lines.append(
                f"| {observation.iteration} | {observation.tokens} | "
                f"{observation.quality:.4f} | {observation.objective:.2f} |"
            )
        lines.extend(
            [
                "",
                "## Pareto Front",
                "",
                "| Iteration | Tokens | Quality | Objective |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for observation in self.pareto_front():
            lines.append(
                f"| {observation.iteration} | {observation.tokens} | "
                f"{observation.quality:.4f} | {observation.objective:.2f} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_html_report(self, path: Path) -> None:
        """Write a standalone HTML report with tables and a simple SVG chart."""

        savings = self.savings_summary()
        chart = _svg_scatter(self.observations, self.best, self.baseline)
        top_rows = "\n".join(_html_observation_row(item) for item in sorted(self.observations, key=lambda item: item.objective)[:10])
        pareto_rows = "\n".join(_html_observation_row(item) for item in self.pareto_front())
        task_rows = "\n".join(_html_task_row(task) for task in self.best.metadata.get("tasks", []))
        task_table = (
            f"<table><thead><tr><th>Task</th><th>Kind</th><th>Tokens</th><th>Quality</th></tr></thead><tbody>{task_rows}</tbody></table>"
            if task_rows
            else "<p>No task-level breakdown available.</p>"
        )
        html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Token Optimization Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d8dee9; padding: 12px; background: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #eef2f7; }}
    code, pre {{ background: #f5f7fa; padding: 8px; display: block; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Token Optimization Report</h1>
  <section class="metrics">
    <div class="metric"><strong>Baseline tokens</strong><br>{self.baseline.tokens if self.baseline else "n/a"}</div>
    <div class="metric"><strong>Optimized tokens</strong><br>{self.best.tokens}</div>
    <div class="metric"><strong>Tokens saved</strong><br>{savings["tokens_saved"]}</div>
    <div class="metric"><strong>Reduction</strong><br>{savings["token_reduction_percent"]}%</div>
    <div class="metric"><strong>Quality</strong><br>{self.best.quality:.4f}</div>
  </section>
  <h2>Token / Quality Landscape</h2>
  {chart}
  <h2>Best Configuration</h2>
  <pre>{html.escape(json.dumps(self.best.config, indent=2, sort_keys=True))}</pre>
  <h2>Best Task Breakdown</h2>
  {task_table}
  <h2>Top Evaluations</h2>
  <table><thead><tr><th>Iteration</th><th>Tokens</th><th>Quality</th><th>Objective</th></tr></thead><tbody>{top_rows}</tbody></table>
  <h2>Pareto Front</h2>
  <table><thead><tr><th>Iteration</th><th>Tokens</th><th>Quality</th><th>Objective</th></tr></thead><tbody>{pareto_rows}</tbody></table>
</body>
</html>
"""
        path.write_text(html_text, encoding="utf-8")


@dataclass(frozen=True)
class TokenOptimizationConfig:
    """Optimization policy."""

    iterations: int = 30
    initial_points: int = 8
    candidate_pool: int = 400
    quality_floor: float = 0.82
    quality_penalty: float = 30000.0
    random_state: int = 42
    evaluate_baseline: bool = True
    acquisition: str = "expected_improvement"
    exploration_weight: float = 1.25
    min_candidate_distance: float = 0.04


@dataclass
class TokenBayesOptimizer:
    """Bayesian optimizer for token usage reduction."""

    search_space: SearchSpace
    objective: Objective
    config: TokenOptimizationConfig = TokenOptimizationConfig()

    def run(self) -> OptimizationResult:
        rng = random.Random(self.config.random_state)
        observations: list[Observation] = []
        baseline = self._evaluate(-1, default_baseline_config()) if self.config.evaluate_baseline else None
        for iteration in range(self.config.iterations):
            if iteration < self.config.initial_points or len(observations) < 3:
                candidate = self.search_space.sample(rng)
            else:
                candidate = self._suggest(observations, rng)
            observations.append(self._evaluate(iteration, candidate))
        feasible = [
            observation
            for observation in observations
            if observation.quality >= float(observation.metadata.get("quality_floor", 0.0))
        ]
        best_pool = feasible or observations
        return OptimizationResult(
            best=min(best_pool, key=lambda item: item.objective),
            observations=observations,
            baseline=baseline,
            optimization_config=self.config,
        )

    def _evaluate(self, iteration: int, candidate: Config) -> Observation:
        objective, metadata = self.objective(candidate)
        return Observation(
            iteration=iteration,
            config=candidate,
            objective=objective,
            tokens=int(metadata["tokens"]),
            quality=float(metadata["quality"]),
            latency_ms=float(metadata.get("latency_ms", 0.0)),
            metadata=metadata,
        )

    def _suggest(self, observations: Sequence[Observation], rng: random.Random) -> Config:
        if self.config.acquisition == "random_search":
            return self.search_space.sample(rng)
        x_train = [self.search_space.encode(observation.config) for observation in observations]
        y_train = [observation.objective for observation in observations]
        model = GaussianProcessRegressor.fit(x_train, y_train)
        best = min(y_train)
        candidates = [self.search_space.sample(rng) for _ in range(self.config.candidate_pool)]
        return max(
            candidates,
            key=lambda config: self._acquisition_score(config, model, best, x_train),
        )

    def _acquisition_score(
        self,
        config: Config,
        model: GaussianProcessRegressor,
        best: float,
        x_train: Sequence[Sequence[float]],
    ) -> float:
        vector = self.search_space.encode(config)
        prediction = model.predict(vector)
        score = acquisition_score(
            prediction,
            best=best,
            strategy=self.config.acquisition,
            exploration_weight=self.config.exploration_weight,
        )
        distance = min(_euclidean_distance(vector, known) for known in x_train)
        if distance < self.config.min_candidate_distance:
            score *= distance / max(self.config.min_candidate_distance, 1e-9)
        return score


@dataclass(frozen=True)
class TokenQualityObjective:
    """Default simulated objective for prompt compression experiments."""

    quality_floor: float = 0.82
    quality_penalty: float = 30000.0

    def __call__(self, config: Config) -> tuple[float, dict[str, Any]]:
        tokens = estimate_tokens(config)
        quality = estimate_quality(config)
        latency_ms = 180.0 + tokens * 0.42
        shortfall = max(0.0, self.quality_floor - quality)
        objective = tokens + self.quality_penalty * (shortfall + shortfall**2)
        return objective, {
            "tokens": tokens,
            "quality": quality,
            "quality_floor": self.quality_floor,
            "quality_shortfall": shortfall,
            "latency_ms": latency_ms,
        }


@dataclass(frozen=True)
class MultiTaskTokenQualityObjective:
    """Evaluate one configuration across several representative workloads."""

    tasks: tuple[Task, ...]
    quality_floor: float = 0.82
    quality_penalty: float = 30000.0

    def __call__(self, config: Config) -> tuple[float, dict[str, Any]]:
        if not self.tasks:
            raise ValueError("At least one task is required.")

        task_results = [evaluate_task(config, task) for task in self.tasks]
        total_weight = sum(task.quality_weight for task in self.tasks)
        total_tokens = sum(int(result["tokens"]) for result in task_results)
        weighted_quality = sum(
            float(result["quality"]) * task.quality_weight
            for result, task in zip(task_results, self.tasks)
        ) / max(total_weight, 1e-9)
        worst_quality = min(float(result["quality"]) for result in task_results)
        latency_ms = sum(float(result["latency_ms"]) for result in task_results)
        shortfall = max(0.0, self.quality_floor - weighted_quality)
        worst_shortfall = max(0.0, self.quality_floor - worst_quality)
        objective = total_tokens + self.quality_penalty * (shortfall + shortfall**2)
        objective += self.quality_penalty * 0.35 * (worst_shortfall + worst_shortfall**2)
        return objective, {
            "tokens": total_tokens,
            "quality": weighted_quality,
            "worst_task_quality": worst_quality,
            "quality_floor": self.quality_floor,
            "quality_shortfall": shortfall,
            "latency_ms": latency_ms,
            "tasks": task_results,
        }


def load_tasks_jsonl(path: Path) -> tuple[Task, ...]:
    """Load task definitions from a JSONL file."""

    tasks: list[Task] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            try:
                tasks.append(
                    Task(
                        name=str(payload["name"]),
                        kind=str(payload.get("kind", "general")),
                        prompt=str(payload["prompt"]),
                        quality_weight=float(payload.get("quality_weight", 1.0)),
                        token_multiplier=float(payload.get("token_multiplier", 1.0)),
                        requires_retrieval=bool(payload.get("requires_retrieval", False)),
                        requires_examples=bool(payload.get("requires_examples", False)),
                        requires_reasoning=bool(payload.get("requires_reasoning", False)),
                    )
                )
            except KeyError as error:
                raise ValueError(f"Task line {line_number} is missing required field: {error}") from error
    if not tasks:
        raise ValueError("Task file must contain at least one task.")
    return tuple(tasks)


def evaluate_task(config: Config, task: Task) -> dict[str, Any]:
    """Simulate token and quality metrics for one task."""

    base_tokens = estimate_tokens(config)
    prompt_tokens = max(24, len(task.prompt) // 4)
    kind_factor = {
        "qa": 0.92,
        "summarization": 1.18,
        "rag": 1.28,
        "code": 1.12,
        "extraction": 0.86,
    }.get(task.kind, 1.0)
    tokens = int((base_tokens * kind_factor + prompt_tokens) * task.token_multiplier)
    quality = estimate_quality(config)
    if task.requires_retrieval:
        quality += min(0.10, float(config["retrieval_top_k"]) * 0.012)
        quality -= 0.05 if int(config["retrieval_top_k"]) <= 2 else 0.0
    if task.requires_examples:
        quality += min(0.08, float(config["few_shot_examples"]) * 0.018)
        quality -= 0.05 if int(config["few_shot_examples"]) == 0 else 0.0
    if task.requires_reasoning:
        quality += {"minimal": -0.06, "low": 0.02, "medium": 0.07}[str(config["reasoning_level"])]
    if task.kind == "summarization":
        quality += 0.04 if 0.35 <= float(config["summary_ratio"]) <= 0.70 else -0.04
    if task.kind == "extraction":
        quality += 0.03 if str(config["format_style"]) in {"compact", "balanced"} else -0.03
    quality = max(0.0, min(1.0, quality))
    return {
        "name": task.name,
        "kind": task.kind,
        "tokens": tokens,
        "quality": quality,
        "latency_ms": 120.0 + tokens * 0.38,
    }


@dataclass(frozen=True)
class GaussianProcessRegressor:
    """Tiny GP regressor with RBF kernel and normalized targets."""

    x_train: list[list[float]]
    y_mean: float
    y_std: float
    alpha: list[float]
    inverse_kernel: list[list[float]]
    length_scale: float = 0.55
    noise: float = 1e-6

    @classmethod
    def fit(cls, x_train: list[list[float]], y_train: list[float]) -> GaussianProcessRegressor:
        y_mean = sum(y_train) / len(y_train)
        variance = sum((value - y_mean) ** 2 for value in y_train) / len(y_train)
        y_std = math.sqrt(variance) or 1.0
        y_scaled = [(value - y_mean) / y_std for value in y_train]
        kernel = [
            [
                rbf_kernel(left, right, length_scale=0.55) + (1e-6 if row == column else 0.0)
                for column, right in enumerate(x_train)
            ]
            for row, left in enumerate(x_train)
        ]
        inverse_kernel = invert_matrix(kernel)
        alpha = mat_vec(inverse_kernel, y_scaled)
        return cls(x_train=x_train, y_mean=y_mean, y_std=y_std, alpha=alpha, inverse_kernel=inverse_kernel)

    def predict(self, x: list[float]) -> tuple[float, float]:
        kernel_vector = [rbf_kernel(x, known, length_scale=self.length_scale) for known in self.x_train]
        mean_scaled = dot(kernel_vector, self.alpha)
        variance_scaled = max(1e-9, 1.0 - dot(kernel_vector, mat_vec(self.inverse_kernel, kernel_vector)))
        return self.y_mean + mean_scaled * self.y_std, math.sqrt(variance_scaled) * self.y_std


def estimate_tokens(config: Config) -> int:
    context_tokens = float(config["max_context_chars"]) / 4.0
    retrieval_tokens = float(config["retrieval_top_k"]) * 95.0
    summary_tokens = context_tokens * float(config["summary_ratio"])
    examples_tokens = float(config["few_shot_examples"]) * 145.0
    reasoning_tokens = {"minimal": 80.0, "low": 180.0, "medium": 360.0}[str(config["reasoning_level"])]
    format_tokens = {"compact": 40.0, "balanced": 95.0, "verbose": 210.0}[str(config["format_style"])]
    return int(220.0 + summary_tokens + retrieval_tokens + examples_tokens + reasoning_tokens + format_tokens)


def estimate_quality(config: Config) -> float:
    context = math.log1p(float(config["max_context_chars"])) / math.log1p(8000.0)
    retrieval = 1.0 - math.exp(-float(config["retrieval_top_k"]) / 4.0)
    summary = 1.0 - abs(float(config["summary_ratio"]) - 0.55) * 0.65
    examples = min(1.0, float(config["few_shot_examples"]) / 3.0)
    reasoning = {"minimal": 0.58, "low": 0.78, "medium": 0.92}[str(config["reasoning_level"])]
    style = {"compact": 0.82, "balanced": 0.94, "verbose": 0.90}[str(config["format_style"])]
    quality = 0.20 * context + 0.22 * retrieval + 0.18 * summary + 0.12 * examples + 0.18 * reasoning + 0.10 * style
    return max(0.0, min(1.0, quality))


def expected_improvement(prediction: tuple[float, float], best: float) -> float:
    mean, std = prediction
    if std <= 1e-9:
        return 0.0
    improvement = best - mean
    z = improvement / std
    return improvement * normal_cdf(z) + std * normal_pdf(z)


def probability_improvement(prediction: tuple[float, float], best: float) -> float:
    """Probability that a candidate improves over the current best value."""

    mean, std = prediction
    if std <= 1e-9:
        return 0.0
    return normal_cdf((best - mean) / std)


def lower_confidence_bound(prediction: tuple[float, float], exploration_weight: float = 1.25) -> float:
    """Lower confidence bound utility converted to a maximization score."""

    mean, std = prediction
    return -(mean - exploration_weight * std)


def acquisition_score(
    prediction: tuple[float, float],
    best: float,
    strategy: str = "expected_improvement",
    exploration_weight: float = 1.25,
) -> float:
    """Score a candidate according to the selected acquisition function."""

    if strategy == "expected_improvement":
        return expected_improvement(prediction, best)
    if strategy == "probability_improvement":
        return probability_improvement(prediction, best)
    if strategy == "lower_confidence_bound":
        return lower_confidence_bound(prediction, exploration_weight=exploration_weight)
    raise ValueError(
        "acquisition must be 'expected_improvement', 'probability_improvement', "
        "'lower_confidence_bound', or 'random_search'."
    )


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


def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _html_observation_row(observation: Observation) -> str:
    return (
        "<tr>"
        f"<td>{observation.iteration}</td>"
        f"<td>{observation.tokens}</td>"
        f"<td>{observation.quality:.4f}</td>"
        f"<td>{observation.objective:.2f}</td>"
        "</tr>"
    )


def _html_task_row(task: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(str(task['name']))}</td>"
        f"<td>{html.escape(str(task['kind']))}</td>"
        f"<td>{int(task['tokens'])}</td>"
        f"<td>{float(task['quality']):.4f}</td>"
        "</tr>"
    )


def _svg_scatter(
    observations: Sequence[Observation],
    best: Observation,
    baseline: Observation | None,
    width: int = 760,
    height: int = 320,
) -> str:
    tokens = [observation.tokens for observation in observations]
    qualities = [observation.quality for observation in observations]
    if baseline is not None:
        tokens.append(baseline.tokens)
        qualities.append(baseline.quality)
    min_tokens, max_tokens = min(tokens), max(tokens)
    min_quality, max_quality = min(qualities), max(qualities)
    padding = 36

    def x_pos(token_value: int) -> float:
        if max_tokens == min_tokens:
            return width / 2
        return padding + (token_value - min_tokens) / (max_tokens - min_tokens) * (width - padding * 2)

    def y_pos(quality_value: float) -> float:
        if max_quality == min_quality:
            return height / 2
        return height - padding - (quality_value - min_quality) / (max_quality - min_quality) * (height - padding * 2)

    circles = []
    for observation in observations:
        color = "#2f80ed" if observation != best else "#219653"
        radius = 4 if observation != best else 7
        circles.append(
            f'<circle cx="{x_pos(observation.tokens):.1f}" cy="{y_pos(observation.quality):.1f}" '
            f'r="{radius}" fill="{color}"><title>iter={observation.iteration}, '
            f'tokens={observation.tokens}, quality={observation.quality:.4f}</title></circle>'
        )
    if baseline is not None:
        circles.append(
            f'<rect x="{x_pos(baseline.tokens) - 5:.1f}" y="{y_pos(baseline.quality) - 5:.1f}" '
            'width="10" height="10" fill="#eb5757"><title>baseline</title></rect>'
        )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Token quality scatter plot">'
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" '
        'stroke="#9aa5b1"/>'
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#9aa5b1"/>'
        f'<text x="{width / 2}" y="{height - 6}" text-anchor="middle" font-size="12">Tokens</text>'
        f'<text x="12" y="{height / 2}" transform="rotate(-90 12 {height / 2})" '
        'text-anchor="middle" font-size="12">Quality</text>'
        + "".join(circles)
        + "</svg>"
    )


def _observation_to_dict(observation: Observation) -> dict[str, Any]:
    return {
        "iteration": observation.iteration,
        "config": observation.config,
        "objective": observation.objective,
        "tokens": observation.tokens,
        "quality": observation.quality,
        "latency_ms": observation.latency_ms,
        "metadata": observation.metadata,
    }


def _config_to_dict(config: TokenOptimizationConfig) -> dict[str, Any]:
    return {
        "iterations": config.iterations,
        "initial_points": config.initial_points,
        "candidate_pool": config.candidate_pool,
        "quality_floor": config.quality_floor,
        "quality_penalty": config.quality_penalty,
        "random_state": config.random_state,
        "evaluate_baseline": config.evaluate_baseline,
        "acquisition": config.acquisition,
        "exploration_weight": config.exploration_weight,
        "min_candidate_distance": config.min_candidate_distance,
    }
