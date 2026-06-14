"""CLI for defensive Bayesian security scan optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

from security_bayes_optimizer.core import BayesianOptimizer, OptimizationConfig, SearchSpace
from security_bayes_optimizer.scanner import SecurityScanObjective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize local defensive static security scan settings.")
    parser.add_argument("--target", type=Path, required=True, help="Authorized local project path to scan.")
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--initial-points", type=int, default=8)
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--acquisition",
        choices=["expected_improvement", "probability_improvement", "lower_confidence_bound", "random_search"],
        default="expected_improvement",
    )
    parser.add_argument("--exploration-weight", type=float, default=1.25)
    parser.add_argument("--json-output", type=Path, default=Path("security_optimization_result.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("security_optimization_history.csv"))
    parser.add_argument("--html-output", type=Path, default=Path("security_optimization_report.html"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.target.exists() or not args.target.is_dir():
        raise SystemExit(f"Target must be an existing local directory: {args.target}")
    config = OptimizationConfig(
        iterations=args.iterations,
        initial_points=args.initial_points,
        candidate_pool=args.candidate_pool,
        random_state=args.random_state,
        acquisition=args.acquisition,
        exploration_weight=args.exploration_weight,
    )
    optimizer = BayesianOptimizer(
        search_space=SearchSpace.security_scan_space(),
        objective=SecurityScanObjective(args.target),
        config=config,
    )
    result = optimizer.run()
    args.json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    result.write_csv(args.csv_output)
    result.write_html_report(args.html_output)
    print("Best defensive scan configuration")
    print(f"  findings: {result.best.findings}")
    print(f"  high findings: {result.best.high_findings}")
    print(f"  precision proxy: {result.best.precision_proxy:.3f}")
    print(f"  elapsed ms: {result.best.elapsed_ms:.2f}")
    print(f"  config: {result.best.config}")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.html_output}")


if __name__ == "__main__":
    main()
