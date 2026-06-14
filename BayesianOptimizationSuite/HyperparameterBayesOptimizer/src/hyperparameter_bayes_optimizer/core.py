"""Hyperparameter tuning adapter built on BayesCore."""

from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayes_core import (
    Config,
    Constraint,
    GenericBayesianOptimizer,
    Objective,
    OptimizationConfig,
    Parameter,
    SearchSpace,
)


class HyperparameterSearchSpace(SearchSpace):
    """Presets for common ML hyperparameter tuning tasks."""

    @classmethod
    def tabular_classification_space(cls) -> HyperparameterSearchSpace:
        return cls(
            parameters=(
                Parameter("learning_rate", "float", low=0.001, high=0.30),
                Parameter("max_depth", "int", low=2, high=12),
                Parameter("n_estimators", "int", low=50, high=500),
                Parameter("regularization", "float", low=0.0, high=2.0),
                Parameter("subsample", "float", low=0.55, high=1.0),
                Parameter("model_family", "categorical", choices=("random_forest", "gradient_boosting", "linear")),
                Parameter("decision_threshold", "float", low=0.25, high=0.75),
            )
        )


@dataclass(frozen=True)
class HyperparameterObservation:
    iteration: int
    config: Config
    objective: float
    f1: float
    accuracy: float
    latency_ms: float
    training_cost: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HyperparameterOptimizationResult:
    best: HyperparameterObservation
    observations: list[HyperparameterObservation]
    optimization_config: OptimizationConfig

    def to_json(self) -> str:
        return json.dumps(
            {
                "best": observation_to_dict(self.best),
                "observations": [observation_to_dict(item) for item in self.observations],
                "optimization_config": self.optimization_config.__dict__,
                "pareto_front": [observation_to_dict(item) for item in self.pareto_front()],
            },
            indent=2,
            sort_keys=True,
        )

    def pareto_front(self) -> list[HyperparameterObservation]:
        front: list[HyperparameterObservation] = []
        for candidate in self.observations:
            dominated = any(
                other.f1 >= candidate.f1
                and other.latency_ms <= candidate.latency_ms
                and other.training_cost <= candidate.training_cost
                and (
                    other.f1 > candidate.f1
                    or other.latency_ms < candidate.latency_ms
                    or other.training_cost < candidate.training_cost
                )
                for other in self.observations
            )
            if not dominated:
                front.append(candidate)
        return sorted(front, key=lambda item: (-item.f1, item.latency_ms, item.training_cost))

    def write_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "iteration",
                    "objective",
                    "f1",
                    "accuracy",
                    "latency_ms",
                    "training_cost",
                    "objective_std",
                    "repeats",
                    "config_json",
                ],
            )
            writer.writeheader()
            for observation in self.observations:
                writer.writerow(
                    {
                        "iteration": observation.iteration,
                        "objective": f"{observation.objective:.6f}",
                        "f1": f"{observation.f1:.6f}",
                        "accuracy": f"{observation.accuracy:.6f}",
                        "latency_ms": f"{observation.latency_ms:.3f}",
                        "training_cost": f"{observation.training_cost:.6f}",
                        "objective_std": f"{float(observation.metadata.get('objective_std', 0.0)):.6f}",
                        "repeats": observation.metadata.get("repeats", 1),
                        "config_json": json.dumps(observation.config, sort_keys=True),
                    }
                )

    def write_html_report(self, path: Path) -> None:
        rows = "\n".join(html_observation_row(item) for item in sorted(self.observations, key=lambda item: item.objective)[:15])
        pareto_rows = "\n".join(html_observation_row(item) for item in self.pareto_front())
        html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hyperparameter Bayesian Optimization Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #182033; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d8dee9; padding: 12px; background: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; font-size: 14px; text-align: left; }}
    th {{ background: #eef2f7; }}
    pre {{ background: #f5f7fa; padding: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Hyperparameter Bayesian Optimization Report</h1>
  <section class="metrics">
    <div class="metric"><strong>F1</strong><br>{self.best.f1:.4f}</div>
    <div class="metric"><strong>Accuracy</strong><br>{self.best.accuracy:.4f}</div>
    <div class="metric"><strong>Latency</strong><br>{self.best.latency_ms:.2f} ms</div>
    <div class="metric"><strong>Training cost</strong><br>{self.best.training_cost:.4f}</div>
  </section>
  <h2>Best Hyperparameters</h2>
  <pre>{html.escape(json.dumps(self.best.config, indent=2, sort_keys=True))}</pre>
  <h2>Top Evaluations</h2>
  <table><thead><tr><th>Iteration</th><th>F1</th><th>Accuracy</th><th>Latency</th><th>Cost</th><th>Objective</th></tr></thead><tbody>{rows}</tbody></table>
  <h2>Pareto Front</h2>
  <table><thead><tr><th>Iteration</th><th>F1</th><th>Accuracy</th><th>Latency</th><th>Cost</th><th>Objective</th></tr></thead><tbody>{pareto_rows}</tbody></table>
</body>
</html>
"""
        path.write_text(html_text, encoding="utf-8")


@dataclass
class HyperparameterBayesOptimizer:
    search_space: HyperparameterSearchSpace
    objective: Objective
    config: OptimizationConfig
    constraints: tuple[Constraint, ...] = ()

    def run(self) -> HyperparameterOptimizationResult:
        generic = GenericBayesianOptimizer(
            search_space=self.search_space,
            objective=self.objective,
            config=self.config,
            constraints=self.constraints,
        ).run()
        observations = [observation_from_generic(item) for item in generic.observations]
        return HyperparameterOptimizationResult(
            best=min(observations, key=lambda item: item.objective),
            observations=observations,
            optimization_config=self.config,
        )


@dataclass
class SimulatedModelObjective:
    """Deterministic demo objective that mimics model validation metrics."""

    min_f1: float = 0.82
    latency_weight: float = 0.0008
    cost_weight: float = 0.03

    def __call__(self, config: Config) -> tuple[float, dict[str, Any]]:
        learning_rate = float(config["learning_rate"])
        max_depth = int(config["max_depth"])
        n_estimators = int(config["n_estimators"])
        regularization = float(config["regularization"])
        subsample = float(config["subsample"])
        threshold = float(config["decision_threshold"])
        family = str(config["model_family"])

        family_bonus = {
            "gradient_boosting": 0.045,
            "random_forest": 0.025,
            "linear": -0.025,
        }[family]
        depth_score = math.exp(-((max_depth - 6) ** 2) / 18)
        lr_score = math.exp(-((math.log10(learning_rate) - math.log10(0.045)) ** 2) / 0.45)
        estimator_score = math.exp(-((n_estimators - 260) ** 2) / 65000)
        reg_score = math.exp(-((regularization - 0.55) ** 2) / 0.55)
        threshold_score = math.exp(-((threshold - 0.48) ** 2) / 0.035)
        subsample_score = 1.0 - abs(subsample - 0.82) * 0.30

        f1 = clamp(
            0.58
            + 0.105 * depth_score
            + 0.090 * lr_score
            + 0.060 * estimator_score
            + 0.045 * reg_score
            + 0.055 * threshold_score
            + 0.035 * subsample_score
            + family_bonus,
            0.0,
            0.97,
        )
        accuracy = clamp(f1 + 0.035 - abs(threshold - 0.50) * 0.08, 0.0, 0.99)
        latency_ms = 12.0 + n_estimators * (0.05 + max_depth * 0.006)
        if family == "gradient_boosting":
            latency_ms *= 1.20
        elif family == "linear":
            latency_ms *= 0.35
        training_cost = (n_estimators / 100.0) * (max_depth / 6.0) * (1.2 if family == "gradient_boosting" else 1.0)

        quality_shortfall = max(0.0, self.min_f1 - f1)
        objective = (1.0 - f1) + self.latency_weight * latency_ms + self.cost_weight * training_cost + 8.0 * quality_shortfall
        return objective, {
            "f1": f1,
            "accuracy": accuracy,
            "latency_ms": latency_ms,
            "training_cost": training_cost,
            "quality_shortfall": quality_shortfall,
            "model_family": family,
        }


def observation_from_generic(observation: Any) -> HyperparameterObservation:
    metadata = observation.metadata
    return HyperparameterObservation(
        iteration=observation.iteration,
        config=observation.config,
        objective=float(observation.objective),
        f1=float(metadata["f1"]),
        accuracy=float(metadata["accuracy"]),
        latency_ms=float(metadata["latency_ms"]),
        training_cost=float(metadata["training_cost"]),
        metadata=metadata,
    )


def observation_to_dict(observation: HyperparameterObservation) -> dict[str, Any]:
    return {
        "iteration": observation.iteration,
        "config": observation.config,
        "objective": observation.objective,
        "f1": observation.f1,
        "accuracy": observation.accuracy,
        "latency_ms": observation.latency_ms,
        "training_cost": observation.training_cost,
        "metadata": observation.metadata,
    }


def html_observation_row(observation: HyperparameterObservation) -> str:
    return (
        "<tr>"
        f"<td>{observation.iteration}</td>"
        f"<td>{observation.f1:.4f}</td>"
        f"<td>{observation.accuracy:.4f}</td>"
        f"<td>{observation.latency_ms:.2f}</td>"
        f"<td>{observation.training_cost:.4f}</td>"
        f"<td>{observation.objective:.4f}</td>"
        "</tr>"
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
