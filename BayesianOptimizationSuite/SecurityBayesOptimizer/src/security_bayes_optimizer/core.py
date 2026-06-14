"""Security-specific adapter around the shared Bayesian optimization core."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayes_core import (
    Config,
    GenericBayesianOptimizer,
    Objective,
    OptimizationConfig,
    Parameter,
    SearchSpace as CoreSearchSpace,
)


class SearchSpace(CoreSearchSpace):
    """Search space presets for defensive local security scanning."""

    @classmethod
    def security_scan_space(cls) -> SearchSpace:
        return cls(
            parameters=(
                Parameter("max_files", "int", low=20, high=500),
                Parameter("max_file_kb", "int", low=32, high=512),
                Parameter("entropy_threshold", "float", low=3.0, high=5.2),
                Parameter("include_tests", "categorical", choices=("yes", "no")),
                Parameter("rule_profile", "categorical", choices=("fast", "balanced", "deep")),
                Parameter("severity_floor", "categorical", choices=("low", "medium", "high")),
            )
        )


@dataclass(frozen=True)
class Observation:
    """One evaluated scan configuration with security-domain metrics."""

    iteration: int
    config: Config
    objective: float
    findings: int
    high_findings: int
    precision_proxy: float
    elapsed_ms: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OptimizationResult:
    """Final security optimization result."""

    best: Observation
    observations: list[Observation]
    optimization_config: OptimizationConfig

    def to_json(self) -> str:
        return json.dumps(
            {
                "best": observation_to_dict(self.best),
                "optimization_config": config_to_dict(self.optimization_config),
                "observations": [observation_to_dict(observation) for observation in self.observations],
                "pareto_front": [observation_to_dict(observation) for observation in self.pareto_front()],
            },
            indent=2,
            sort_keys=True,
        )

    def pareto_front(self) -> list[Observation]:
        front: list[Observation] = []
        for candidate in self.observations:
            dominated = any(
                other.findings >= candidate.findings
                and other.elapsed_ms <= candidate.elapsed_ms
                and other.precision_proxy >= candidate.precision_proxy
                and (
                    other.findings > candidate.findings
                    or other.elapsed_ms < candidate.elapsed_ms
                    or other.precision_proxy > candidate.precision_proxy
                )
                for other in self.observations
            )
            if not dominated:
                front.append(candidate)
        return sorted(front, key=lambda item: (-item.high_findings, -item.findings, item.elapsed_ms))

    def write_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "iteration",
                    "objective",
                    "findings",
                    "high_findings",
                    "precision_proxy",
                    "elapsed_ms",
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
                        "findings": observation.findings,
                        "high_findings": observation.high_findings,
                        "precision_proxy": f"{observation.precision_proxy:.6f}",
                        "elapsed_ms": f"{observation.elapsed_ms:.2f}",
                        "config_json": json.dumps(observation.config, sort_keys=True),
                        "metadata_json": json.dumps(observation.metadata, sort_keys=True),
                    }
                )

    def write_html_report(self, path: Path) -> None:
        rows = "\n".join(
            html_observation_row(item) for item in sorted(self.observations, key=lambda item: item.objective)[:12]
        )
        pareto_rows = "\n".join(html_observation_row(item) for item in self.pareto_front())
        findings_rows = "\n".join(html_finding_row(item) for item in self.best.metadata.get("findings_list", [])[:25])
        html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Security Bayesian Optimization Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d8dee9; padding: 12px; background: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; font-size: 14px; text-align: left; }}
    th {{ background: #eef2f7; }}
    pre {{ background: #f5f7fa; padding: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Security Bayesian Optimization Report</h1>
  <p>Defensive local static analysis only. Findings require manual validation.</p>
  <section class="metrics">
    <div class="metric"><strong>Findings</strong><br>{self.best.findings}</div>
    <div class="metric"><strong>High findings</strong><br>{self.best.high_findings}</div>
    <div class="metric"><strong>Precision proxy</strong><br>{self.best.precision_proxy:.3f}</div>
    <div class="metric"><strong>Elapsed</strong><br>{self.best.elapsed_ms:.2f} ms</div>
  </section>
  <h2>Best Configuration</h2>
  <pre>{html.escape(json.dumps(self.best.config, indent=2, sort_keys=True))}</pre>
  <h2>Best Findings</h2>
  <table><thead><tr><th>Severity</th><th>Rule</th><th>File</th><th>Line</th><th>Message</th></tr></thead><tbody>{findings_rows}</tbody></table>
  <h2>Top Evaluations</h2>
  <table><thead><tr><th>Iteration</th><th>Findings</th><th>High</th><th>Precision</th><th>Elapsed</th><th>Objective</th></tr></thead><tbody>{rows}</tbody></table>
  <h2>Pareto Front</h2>
  <table><thead><tr><th>Iteration</th><th>Findings</th><th>High</th><th>Precision</th><th>Elapsed</th><th>Objective</th></tr></thead><tbody>{pareto_rows}</tbody></table>
</body>
</html>
"""
        path.write_text(html_text, encoding="utf-8")


@dataclass
class BayesianOptimizer:
    """Security optimizer powered by the shared BayesCore engine."""

    search_space: SearchSpace
    objective: Objective
    config: OptimizationConfig = OptimizationConfig(iterations=24)

    def run(self) -> OptimizationResult:
        generic_result = GenericBayesianOptimizer(
            search_space=self.search_space,
            objective=self.objective,
            config=self.config,
        ).run()
        observations = [observation_from_generic(item) for item in generic_result.observations]
        return OptimizationResult(
            best=min(observations, key=lambda item: item.objective),
            observations=observations,
            optimization_config=self.config,
        )


def observation_from_generic(observation: Any) -> Observation:
    metadata = observation.metadata
    return Observation(
        iteration=observation.iteration,
        config=observation.config,
        objective=float(observation.objective),
        findings=int(metadata["findings"]),
        high_findings=int(metadata["high_findings"]),
        precision_proxy=float(metadata["precision_proxy"]),
        elapsed_ms=float(metadata["elapsed_ms"]),
        metadata=metadata,
    )


def observation_to_dict(observation: Observation) -> dict[str, Any]:
    return {
        "iteration": observation.iteration,
        "config": observation.config,
        "objective": observation.objective,
        "findings": observation.findings,
        "high_findings": observation.high_findings,
        "precision_proxy": observation.precision_proxy,
        "elapsed_ms": observation.elapsed_ms,
        "metadata": observation.metadata,
    }


def config_to_dict(config: OptimizationConfig) -> dict[str, Any]:
    return {
        "iterations": config.iterations,
        "initial_points": config.initial_points,
        "candidate_pool": config.candidate_pool,
        "random_state": config.random_state,
        "acquisition": config.acquisition,
        "exploration_weight": config.exploration_weight,
        "min_candidate_distance": config.min_candidate_distance,
    }


def html_observation_row(observation: Observation) -> str:
    return (
        "<tr>"
        f"<td>{observation.iteration}</td>"
        f"<td>{observation.findings}</td>"
        f"<td>{observation.high_findings}</td>"
        f"<td>{observation.precision_proxy:.3f}</td>"
        f"<td>{observation.elapsed_ms:.2f}</td>"
        f"<td>{observation.objective:.2f}</td>"
        "</tr>"
    )


def html_finding_row(finding: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(str(finding['severity']))}</td>"
        f"<td>{html.escape(str(finding['rule_id']))}</td>"
        f"<td>{html.escape(str(finding['path']))}</td>"
        f"<td>{finding['line']}</td>"
        f"<td>{html.escape(str(finding['message']))}</td>"
        "</tr>"
    )
