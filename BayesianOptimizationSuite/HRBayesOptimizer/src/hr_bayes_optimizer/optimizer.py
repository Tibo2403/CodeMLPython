"""HR-specific adapter for the reusable Bayesian optimization core."""

from __future__ import annotations

import csv
import html
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hr_bayes_optimizer.core import (
    BayesianOptimizationConfig,
    BayesianOptimizer,
    Config,
    Constraint,
    GaussianProcessRegressor,
    GenericObservation,
    Objective,
    Parameter,
    SearchSpace as CoreSearchSpace,
    clamp,
    expected_improvement,
)


class SearchSpace(CoreSearchSpace):
    """HR search-space factory plus the generic search-space behavior."""

    @classmethod
    def hr_policy_space(cls) -> SearchSpace:
        """Default search space for recruitment and retention policy tuning."""

        return cls(
            parameters=(
                Parameter("sourcing_budget_eur", "int", low=800, high=8000),
                Parameter("interview_rounds", "int", low=1, high=5),
                Parameter("assessment_weight", "float", low=0.0, high=1.0),
                Parameter("onboarding_hours", "int", low=4, high=80),
                Parameter("remote_days_per_week", "int", low=0, high=5),
                Parameter("referral_bonus_eur", "int", low=0, high=3000),
                Parameter("screening_policy", "categorical", choices=("manual", "balanced", "data_assisted")),
            )
        )


def default_baseline_config() -> Config:
    """Return a conservative but expensive baseline HR policy."""

    return {
        "sourcing_budget_eur": 6500,
        "interview_rounds": 4,
        "assessment_weight": 0.55,
        "onboarding_hours": 48,
        "remote_days_per_week": 2,
        "referral_bonus_eur": 1500,
        "screening_policy": "manual",
    }


@dataclass(frozen=True)
class Observation:
    """One evaluated HR policy."""

    iteration: int
    config: Config
    objective: float
    cost_eur: float
    quality: float
    retention: float
    fairness_gap: float
    time_to_hire_days: float
    wellbeing: float
    objective_std: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OptimizationResult:
    """Final HR optimization output."""

    best: Observation
    observations: list[Observation]
    baseline: Observation | None = None

    @classmethod
    def from_json(cls, text: str) -> OptimizationResult:
        payload = json.loads(text)
        observations = [_observation_from_dict(item) for item in payload.get("observations", [])]
        baseline_payload = payload.get("baseline")
        baseline = _observation_from_dict(baseline_payload) if baseline_payload else None
        best_payload = payload.get("best")
        best = _observation_from_dict(best_payload) if best_payload else _best_observation(observations)
        return cls(best=best, observations=observations, baseline=baseline)

    @classmethod
    def read_json(cls, path: Path) -> OptimizationResult:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def to_json(self) -> str:
        return json.dumps(
            {
                "best": _observation_to_dict(self.best),
                "baseline": _observation_to_dict(self.baseline) if self.baseline else None,
                "savings": self.savings_summary(),
                "observations": [_observation_to_dict(observation) for observation in self.observations],
            },
            indent=2,
            sort_keys=True,
        )

    def savings_summary(self) -> dict[str, float | None]:
        if self.baseline is None:
            return {
                "cost_saved_eur": None,
                "cost_reduction_percent": None,
                "time_to_hire_days_saved": None,
                "quality_delta": None,
                "retention_delta": None,
                "fairness_gap_delta": None,
            }
        cost_saved = self.baseline.cost_eur - self.best.cost_eur
        return {
            "cost_saved_eur": round(cost_saved, 2),
            "cost_reduction_percent": round(cost_saved / self.baseline.cost_eur * 100.0, 2)
            if self.baseline.cost_eur
            else None,
            "time_to_hire_days_saved": round(self.baseline.time_to_hire_days - self.best.time_to_hire_days, 2),
            "quality_delta": round(self.best.quality - self.baseline.quality, 4),
            "retention_delta": round(self.best.retention - self.baseline.retention, 4),
            "fairness_gap_delta": round(self.best.fairness_gap - self.baseline.fairness_gap, 4),
        }

    def pareto_front(self) -> list[Observation]:
        """Return non-dominated observations for cost/quality/fairness trade-offs."""

        front: list[Observation] = []
        for candidate in self.observations:
            dominated = any(
                other.cost_eur <= candidate.cost_eur
                and other.quality >= candidate.quality
                and other.fairness_gap <= candidate.fairness_gap
                and (
                    other.cost_eur < candidate.cost_eur
                    or other.quality > candidate.quality
                    or other.fairness_gap < candidate.fairness_gap
                )
                for other in self.observations
            )
            if not dominated:
                front.append(candidate)
        return sorted(front, key=lambda item: (item.cost_eur, -item.quality, item.fairness_gap))

    def write_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "iteration",
                    "objective",
                    "cost_eur",
                    "quality",
                    "retention",
                    "fairness_gap",
                    "time_to_hire_days",
                    "wellbeing",
                    "objective_std",
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
                        "cost_eur": f"{observation.cost_eur:.2f}",
                        "quality": f"{observation.quality:.6f}",
                        "retention": f"{observation.retention:.6f}",
                        "fairness_gap": f"{observation.fairness_gap:.6f}",
                        "time_to_hire_days": f"{observation.time_to_hire_days:.2f}",
                        "wellbeing": f"{observation.wellbeing:.6f}",
                        "objective_std": f"{observation.objective_std:.6f}",
                        "config_json": json.dumps(observation.config, sort_keys=True),
                        "metadata_json": json.dumps(observation.metadata, sort_keys=True),
                    }
                )

    def write_markdown_report(self, path: Path) -> None:
        constraints = self.best.metadata
        feasible_count = sum(1 for observation in self.observations if is_feasible(observation))
        savings = self.savings_summary()
        lines = [
            "# HR Bayesian Optimization Report",
            "",
            "## Objective",
            "",
            (
                "Minimize HR operating cost and hiring speed penalties while respecting quality, retention, "
                "fairness, and wellbeing constraints."
            ),
            "",
            "## Savings",
            "",
        ]
        if self.baseline is not None:
            lines.extend(
                [
                    f"- Baseline cost: {self.baseline.cost_eur:.2f} EUR",
                    f"- Optimized cost: {self.best.cost_eur:.2f} EUR",
                    f"- Cost saved: {savings['cost_saved_eur']} EUR",
                    f"- Cost reduction: {savings['cost_reduction_percent']}%",
                    f"- Time-to-hire saved: {savings['time_to_hire_days_saved']} days",
                    f"- Quality delta: {savings['quality_delta']}",
                    f"- Retention delta: {savings['retention_delta']}",
                    f"- Fairness gap delta: {savings['fairness_gap_delta']}",
                    "",
                ]
            )
        else:
            lines.extend(["- No baseline was evaluated.", ""])
        lines.extend(
            [
                "## Best Policy",
                "",
                f"- Cost: {self.best.cost_eur:.2f} EUR",
                f"- Quality: {self.best.quality:.4f} (floor: {constraints['quality_floor']})",
                f"- Retention: {self.best.retention:.4f} (floor: {constraints['retention_floor']})",
                f"- Fairness gap: {self.best.fairness_gap:.4f} (max: {constraints['max_fairness_gap']})",
                f"- Time to hire: {self.best.time_to_hire_days:.2f} days",
                f"- Wellbeing: {self.best.wellbeing:.4f} (floor: {constraints['wellbeing_floor']})",
                f"- Objective: {self.best.objective:.4f}",
                f"- Objective std: {self.best.objective_std:.4f}",
                f"- Feasible evaluations: {feasible_count}/{len(self.observations)}",
                "",
                "```json",
                json.dumps(self.best.config, indent=2, sort_keys=True),
                "```",
                "",
                "## Top Evaluations",
                "",
                "| Iteration | Cost EUR | Quality | Retention | Fairness gap | Time days | Objective |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for observation in sorted(self.observations, key=lambda item: item.objective)[:10]:
            lines.append(_markdown_observation_row(observation))
        lines.extend(
            [
                "",
                "## Pareto Front",
                "",
                "| Iteration | Cost EUR | Quality | Retention | Fairness gap | Time days | Objective |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for observation in self.pareto_front():
            lines.append(_markdown_observation_row(observation))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_html_report(self, path: Path) -> None:
        savings = self.savings_summary()
        chart = _svg_scatter(self.observations, self.best, self.baseline)
        top_rows = "\n".join(
            _html_observation_row(item) for item in sorted(self.observations, key=lambda item: item.objective)[:10]
        )
        pareto_rows = "\n".join(_html_observation_row(item) for item in self.pareto_front())
        html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HR Bayesian Optimization Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d8dee9; padding: 12px; background: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #eef2f7; }}
    pre {{ background: #f5f7fa; padding: 8px; display: block; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>HR Bayesian Optimization Report</h1>
  <section class="metrics">
    <div class="metric"><strong>Baseline cost</strong><br>
      {self.baseline.cost_eur if self.baseline else "n/a"} EUR
    </div>
    <div class="metric"><strong>Optimized cost</strong><br>{self.best.cost_eur:.2f} EUR</div>
    <div class="metric"><strong>Cost saved</strong><br>{savings["cost_saved_eur"]} EUR</div>
    <div class="metric"><strong>Reduction</strong><br>{savings["cost_reduction_percent"]}%</div>
    <div class="metric"><strong>Quality</strong><br>{self.best.quality:.4f}</div>
    <div class="metric"><strong>Fairness gap</strong><br>{self.best.fairness_gap:.4f}</div>
  </section>
  <h2>Cost / Quality Landscape</h2>
  {chart}
  <h2>Best Policy</h2>
  <pre>{html.escape(json.dumps(self.best.config, indent=2, sort_keys=True))}</pre>
  <h2>Top Evaluations</h2>
  <table>
    <thead>
      <tr>
        <th>Iteration</th><th>Cost EUR</th><th>Quality</th><th>Retention</th>
        <th>Fairness gap</th><th>Time days</th><th>Objective</th>
      </tr>
    </thead>
    <tbody>{top_rows}</tbody>
  </table>
  <h2>Pareto Front</h2>
  <table>
    <thead>
      <tr>
        <th>Iteration</th><th>Cost EUR</th><th>Quality</th><th>Retention</th>
        <th>Fairness gap</th><th>Time days</th><th>Objective</th>
      </tr>
    </thead>
    <tbody>{pareto_rows}</tbody>
  </table>
</body>
</html>
"""
        path.write_text(html_text, encoding="utf-8")


@dataclass(frozen=True)
class HROptimizationConfig(BayesianOptimizationConfig):
    """HR optimization policy and business constraints."""

    quality_floor: float = 0.76
    retention_floor: float = 0.72
    wellbeing_floor: float = 0.68
    max_fairness_gap: float = 0.10
    constraint_penalty: float = 25000.0


@dataclass
class HRBayesOptimizer(BayesianOptimizer):
    """Bayesian optimizer for HR policy tuning."""

    search_space: CoreSearchSpace
    objective: Objective
    config: HROptimizationConfig = HROptimizationConfig()
    baseline_config: Config | None = None

    def __post_init__(self) -> None:
        if self.baseline_config is None:
            self.baseline_config = default_baseline_config()
        if not self.constraints:
            self.constraints = (
                Constraint("quality", "min", self.config.quality_floor),
                Constraint("retention", "min", self.config.retention_floor),
                Constraint("wellbeing", "min", self.config.wellbeing_floor),
                Constraint("fairness_gap", "max", self.config.max_fairness_gap),
            )

    def run(
        self,
        initial_observations: list[Observation] | None = None,
        baseline: Observation | None = None,
    ) -> OptimizationResult:
        result = super().run(initial_observations=initial_observations, baseline=baseline)
        return OptimizationResult(
            best=result.best,
            observations=result.observations,
            baseline=result.baseline,
        )

    def _make_observation(
        self,
        iteration: int,
        candidate: Config,
        objective: float,
        metadata: dict[str, Any],
    ) -> Observation:
        return Observation(
            iteration=iteration,
            config=candidate,
            objective=objective,
            cost_eur=float(metadata["cost_eur"]),
            quality=float(metadata["quality"]),
            retention=float(metadata["retention"]),
            fairness_gap=float(metadata["fairness_gap"]),
            time_to_hire_days=float(metadata["time_to_hire_days"]),
            wellbeing=float(metadata["wellbeing"]),
            objective_std=float(metadata.get("objective_std", 0.0)),
            metadata=metadata,
        )

    def _is_feasible(self, observation: GenericObservation) -> bool:
        return is_feasible(observation)


@dataclass(frozen=True)
class HRPolicyObjective:
    """Default simulated objective for HR optimization experiments."""

    quality_floor: float = 0.76
    retention_floor: float = 0.72
    wellbeing_floor: float = 0.68
    max_fairness_gap: float = 0.10
    constraint_penalty: float = 25000.0
    metric_noise: float = 0.0
    cost_noise_eur: float = 0.0
    random_state: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rng", random.Random(self.random_state))

    def __call__(self, config: Config) -> tuple[float, dict[str, Any]]:
        metrics = simulate_hr_metrics(config)
        metrics = self._add_noise(metrics)
        quality_shortfall = max(0.0, self.quality_floor - metrics["quality"])
        retention_shortfall = max(0.0, self.retention_floor - metrics["retention"])
        wellbeing_shortfall = max(0.0, self.wellbeing_floor - metrics["wellbeing"])
        fairness_excess = max(0.0, metrics["fairness_gap"] - self.max_fairness_gap)
        constraint_loss = quality_shortfall + retention_shortfall + wellbeing_shortfall + fairness_excess * 1.5
        objective = (
            metrics["cost_eur"]
            + metrics["time_to_hire_days"] * 110.0
            + self.constraint_penalty * (constraint_loss + constraint_loss**2)
        )
        return objective, {
            **metrics,
            "quality_floor": self.quality_floor,
            "retention_floor": self.retention_floor,
            "wellbeing_floor": self.wellbeing_floor,
            "max_fairness_gap": self.max_fairness_gap,
            "quality_shortfall": quality_shortfall,
            "retention_shortfall": retention_shortfall,
            "wellbeing_shortfall": wellbeing_shortfall,
            "fairness_excess": fairness_excess,
            "metric_noise": self.metric_noise,
            "cost_noise_eur": self.cost_noise_eur,
        }

    def _add_noise(self, metrics: dict[str, float]) -> dict[str, float]:
        if self.metric_noise <= 0.0 and self.cost_noise_eur <= 0.0:
            return metrics
        rng = self._rng
        noisy = dict(metrics)
        for name in ("quality", "retention", "wellbeing"):
            noisy[name] = clamp(metrics[name] + rng.gauss(0.0, self.metric_noise), 0.0, 1.0)
        noisy["fairness_gap"] = clamp(metrics["fairness_gap"] + rng.gauss(0.0, self.metric_noise / 2.0), 0.0, 1.0)
        noisy["time_to_hire_days"] = max(5.0, metrics["time_to_hire_days"] + rng.gauss(0.0, self.metric_noise * 14.0))
        noisy["cost_eur"] = max(0.0, metrics["cost_eur"] + rng.gauss(0.0, self.cost_noise_eur))
        return {key: round(value, 6) for key, value in noisy.items()}


def simulate_hr_metrics(config: Config) -> dict[str, float]:
    """Return deterministic HR metrics for a candidate policy."""

    sourcing_budget = float(config["sourcing_budget_eur"])
    interview_rounds = float(config["interview_rounds"])
    assessment_weight = float(config["assessment_weight"])
    onboarding_hours = float(config["onboarding_hours"])
    remote_days = float(config["remote_days_per_week"])
    referral_bonus = float(config["referral_bonus_eur"])
    screening_policy = str(config["screening_policy"])

    policy_quality = {"manual": 0.61, "balanced": 0.72, "data_assisted": 0.76}[screening_policy]
    policy_speed = {"manual": 4.5, "balanced": 1.8, "data_assisted": 0.4}[screening_policy]
    policy_fairness = {"manual": 0.055, "balanced": 0.04, "data_assisted": 0.085}[screening_policy]

    sourcing_effect = 1.0 - pow(2.718281828459045, -sourcing_budget / 3600.0)
    referral_effect = 1.0 - pow(2.718281828459045, -referral_bonus / 1400.0)
    interview_fit = pow(2.718281828459045, -((interview_rounds - 3.0) ** 2) / 3.0)
    assessment_fit = pow(2.718281828459045, -((assessment_weight - 0.58) ** 2) / 0.11)
    onboarding_effect = 1.0 - pow(2.718281828459045, -onboarding_hours / 28.0)
    remote_fit = pow(2.718281828459045, -((remote_days - 2.8) ** 2) / 4.8)

    quality = (
        0.18 * sourcing_effect
        + 0.12 * referral_effect
        + 0.18 * interview_fit
        + 0.18 * assessment_fit
        + 0.12 * onboarding_effect
        + 0.08 * remote_fit
        + 0.14 * policy_quality
    )
    retention = (
        0.28 * onboarding_effect + 0.25 * remote_fit + 0.18 * quality + 0.12 * referral_effect + 0.17
    )
    wellbeing = (
        0.36 * remote_fit
        + 0.22 * onboarding_effect
        + 0.18 * (1.0 - max(0.0, interview_rounds - 3.0) / 2.0)
        + 0.18
    )
    fairness_gap = max(
        0.015,
        policy_fairness
        + abs(assessment_weight - 0.50) * 0.055
        + max(0.0, 1800.0 - sourcing_budget) / 50000.0
        - min(onboarding_hours, 60.0) / 2200.0,
    )
    time_to_hire_days = (
        34.0
        + interview_rounds * 3.1
        - sourcing_effect * 5.5
        - referral_effect * 3.5
        - policy_speed
        + max(0.0, assessment_weight - 0.7) * 5.0
    )
    cost_eur = (
        950.0
        + sourcing_budget
        + referral_bonus * 0.42
        + interview_rounds * 420.0
        + onboarding_hours * 52.0
        + {"manual": 850.0, "balanced": 620.0, "data_assisted": 980.0}[screening_policy]
    )
    return {
        "cost_eur": round(cost_eur, 2),
        "quality": round(max(0.0, min(1.0, quality)), 6),
        "retention": round(max(0.0, min(1.0, retention)), 6),
        "fairness_gap": round(max(0.0, min(1.0, fairness_gap)), 6),
        "time_to_hire_days": round(max(5.0, time_to_hire_days), 2),
        "wellbeing": round(max(0.0, min(1.0, wellbeing)), 6),
    }


def is_feasible(observation: GenericObservation) -> bool:
    metadata = observation.metadata
    return (
        float(metadata["quality"]) >= float(metadata.get("quality_floor", 0.0))
        and float(metadata["retention"]) >= float(metadata.get("retention_floor", 0.0))
        and float(metadata["wellbeing"]) >= float(metadata.get("wellbeing_floor", 0.0))
        and float(metadata["fairness_gap"]) <= float(metadata.get("max_fairness_gap", 1.0))
    )


def _markdown_observation_row(observation: Observation) -> str:
    return (
        f"| {observation.iteration} | {observation.cost_eur:.2f} | {observation.quality:.4f} | "
        f"{observation.retention:.4f} | {observation.fairness_gap:.4f} | "
        f"{observation.time_to_hire_days:.2f} | {observation.objective:.2f} |"
    )


def _html_observation_row(observation: Observation) -> str:
    return (
        "<tr>"
        f"<td>{observation.iteration}</td>"
        f"<td>{observation.cost_eur:.2f}</td>"
        f"<td>{observation.quality:.4f}</td>"
        f"<td>{observation.retention:.4f}</td>"
        f"<td>{observation.fairness_gap:.4f}</td>"
        f"<td>{observation.time_to_hire_days:.2f}</td>"
        f"<td>{observation.objective:.2f}</td>"
        "</tr>"
    )


def _svg_scatter(
    observations: list[Observation],
    best: Observation,
    baseline: Observation | None,
    width: int = 760,
    height: int = 320,
) -> str:
    costs = [observation.cost_eur for observation in observations]
    qualities = [observation.quality for observation in observations]
    if baseline is not None:
        costs.append(baseline.cost_eur)
        qualities.append(baseline.quality)
    min_cost, max_cost = min(costs), max(costs)
    min_quality, max_quality = min(qualities), max(qualities)
    padding = 36

    def x_pos(cost_value: float) -> float:
        if max_cost == min_cost:
            return width / 2
        return padding + (cost_value - min_cost) / (max_cost - min_cost) * (width - padding * 2)

    def y_pos(quality_value: float) -> float:
        if max_quality == min_quality:
            return height / 2
        return height - padding - (quality_value - min_quality) / (max_quality - min_quality) * (height - padding * 2)

    points = []
    for observation in observations:
        color = "#2f80ed" if observation != best else "#219653"
        radius = 4 if observation != best else 7
        points.append(
            f'<circle cx="{x_pos(observation.cost_eur):.1f}" cy="{y_pos(observation.quality):.1f}" '
            f'r="{radius}" fill="{color}"><title>iter={observation.iteration}, '
            f'cost={observation.cost_eur:.2f}, quality={observation.quality:.4f}</title></circle>'
        )
    if baseline is not None:
        points.append(
            f'<rect x="{x_pos(baseline.cost_eur) - 5:.1f}" y="{y_pos(baseline.quality) - 5:.1f}" '
            'width="10" height="10" fill="#eb5757"><title>baseline</title></rect>'
        )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Cost quality scatter plot">'
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" '
        'stroke="#9aa5b1"/>'
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#9aa5b1"/>'
        f'<text x="{width / 2}" y="{height - 6}" text-anchor="middle" font-size="12">Cost EUR</text>'
        f'<text x="12" y="{height / 2}" transform="rotate(-90 12 {height / 2})" '
        'text-anchor="middle" font-size="12">Quality</text>'
        + "".join(points)
        + "</svg>"
    )


def _observation_to_dict(observation: Observation) -> dict[str, Any]:
    return {
        "iteration": observation.iteration,
        "config": observation.config,
        "objective": observation.objective,
        "cost_eur": observation.cost_eur,
        "quality": observation.quality,
        "retention": observation.retention,
        "fairness_gap": observation.fairness_gap,
        "time_to_hire_days": observation.time_to_hire_days,
        "wellbeing": observation.wellbeing,
        "objective_std": observation.objective_std,
        "metadata": observation.metadata,
    }


def _observation_from_dict(payload: dict[str, Any]) -> Observation:
    return Observation(
        iteration=int(payload["iteration"]),
        config=dict(payload["config"]),
        objective=float(payload["objective"]),
        cost_eur=float(payload["cost_eur"]),
        quality=float(payload["quality"]),
        retention=float(payload["retention"]),
        fairness_gap=float(payload["fairness_gap"]),
        time_to_hire_days=float(payload["time_to_hire_days"]),
        wellbeing=float(payload["wellbeing"]),
        objective_std=float(payload.get("objective_std", payload.get("metadata", {}).get("objective_std", 0.0))),
        metadata=dict(payload.get("metadata", {})),
    )


def _best_observation(observations: list[Observation]) -> Observation:
    if not observations:
        raise ValueError("Cannot load optimization result without observations.")
    feasible = [observation for observation in observations if is_feasible(observation)]
    return min(feasible or observations, key=lambda item: item.objective)
