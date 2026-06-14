"""Command-line interface for HR Bayesian optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

from hr_bayes_optimizer.optimizer import (
    HRBayesOptimizer,
    HROptimizationConfig,
    HRPolicyObjective,
    OptimizationResult,
    SearchSpace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize HR policy parameters with Bayesian optimization.")
    parser.add_argument("--iterations", type=int, default=35, help="Number of optimization evaluations.")
    parser.add_argument("--initial-points", type=int, default=8, help="Random warm-up evaluations.")
    parser.add_argument("--quality-floor", type=float, default=0.76, help="Minimum acceptable hiring quality.")
    parser.add_argument("--retention-floor", type=float, default=0.72, help="Minimum acceptable retention score.")
    parser.add_argument("--wellbeing-floor", type=float, default=0.68, help="Minimum acceptable wellbeing score.")
    parser.add_argument("--max-fairness-gap", type=float, default=0.10, help="Maximum acceptable fairness gap.")
    parser.add_argument("--constraint-penalty", type=float, default=25000.0, help="Penalty for constraint violations.")
    parser.add_argument(
        "--observation-repeats",
        type=int,
        default=1,
        help="Repeat each policy evaluation and average it.",
    )
    parser.add_argument("--gp-noise", type=float, default=1e-4, help="Minimum GP observation noise variance.")
    parser.add_argument("--exploration", type=float, default=0.01, help="Expected Improvement exploration margin.")
    parser.add_argument(
        "--unconstrained-acquisition",
        action="store_true",
        help="Use raw Expected Improvement instead of EI multiplied by feasibility probability.",
    )
    parser.add_argument(
        "--metric-noise",
        type=float,
        default=0.0,
        help="Demo simulator noise added to bounded metrics.",
    )
    parser.add_argument("--cost-noise-eur", type=float, default=0.0, help="Demo simulator noise added to cost.")
    parser.add_argument("--random-state", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--no-baseline", action="store_true", help="Do not evaluate the baseline policy.")
    parser.add_argument("--json-output", type=Path, default=Path("hr_optimization_result.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("hr_optimization_history.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("hr_optimization_report.md"))
    parser.add_argument("--html-output", type=Path, default=Path("hr_optimization_report.html"))
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from a previous hr_optimization_result.json file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = HROptimizationConfig(
        iterations=args.iterations,
        initial_points=args.initial_points,
        quality_floor=args.quality_floor,
        retention_floor=args.retention_floor,
        wellbeing_floor=args.wellbeing_floor,
        max_fairness_gap=args.max_fairness_gap,
        constraint_penalty=args.constraint_penalty,
        observation_repeats=args.observation_repeats,
        gp_noise=args.gp_noise,
        exploration=args.exploration,
        use_constrained_acquisition=not args.unconstrained_acquisition,
        random_state=args.random_state,
        evaluate_baseline=not args.no_baseline,
    )
    objective = HRPolicyObjective(
        quality_floor=policy.quality_floor,
        retention_floor=policy.retention_floor,
        wellbeing_floor=policy.wellbeing_floor,
        max_fairness_gap=policy.max_fairness_gap,
        constraint_penalty=policy.constraint_penalty,
        metric_noise=args.metric_noise,
        cost_noise_eur=args.cost_noise_eur,
        random_state=args.random_state,
    )
    optimizer = HRBayesOptimizer(
        search_space=SearchSpace.hr_policy_space(),
        objective=objective,
        config=policy,
    )
    previous = OptimizationResult.read_json(args.resume_from) if args.resume_from else None
    result = optimizer.run(
        initial_observations=previous.observations if previous else None,
        baseline=previous.baseline if previous else None,
    )
    args.json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    result.write_csv(args.csv_output)
    result.write_markdown_report(args.report_output)
    result.write_html_report(args.html_output)
    print("Best HR policy")
    print(f"  cost_eur: {result.best.cost_eur:.2f}")
    print(f"  quality: {result.best.quality:.4f}")
    print(f"  retention: {result.best.retention:.4f}")
    print(f"  fairness_gap: {result.best.fairness_gap:.4f}")
    print(f"  time_to_hire_days: {result.best.time_to_hire_days:.2f}")
    print(f"  wellbeing: {result.best.wellbeing:.4f}")
    print(f"  objective: {result.best.objective:.4f}")
    if result.baseline is not None:
        savings = result.savings_summary()
        print(f"  baseline_cost_eur: {result.baseline.cost_eur:.2f}")
        print(f"  cost_saved_eur: {savings['cost_saved_eur']} ({savings['cost_reduction_percent']}%)")
    if previous is not None:
        print(f"  resumed_from: {args.resume_from}")
        print(f"  previous_observations: {len(previous.observations)}")
        print(f"  total_observations: {len(result.observations)}")
    print(f"  config: {result.best.config}")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.report_output}")
    print(f"Wrote {args.html_output}")


if __name__ == "__main__":
    main()
