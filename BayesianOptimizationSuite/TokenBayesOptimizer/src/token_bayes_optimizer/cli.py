"""Command-line interface for token Bayesian optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

from token_bayes_optimizer.optimizer import (
    MultiTaskTokenQualityObjective,
    SearchSpace,
    TokenBayesOptimizer,
    TokenOptimizationConfig,
    TokenQualityObjective,
    load_tasks_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize AI prompt settings to reduce token usage.")
    parser.add_argument("--iterations", type=int, default=30, help="Number of optimization evaluations.")
    parser.add_argument("--initial-points", type=int, default=8, help="Random warm-up evaluations.")
    parser.add_argument("--quality-floor", type=float, default=0.82, help="Minimum acceptable quality score.")
    parser.add_argument("--quality-penalty", type=float, default=30000.0, help="Penalty for missing quality floor.")
    parser.add_argument("--random-state", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument(
        "--acquisition",
        choices=["expected_improvement", "probability_improvement", "lower_confidence_bound"],
        default="expected_improvement",
        help="Bayesian acquisition strategy.",
    )
    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=1.25,
        help="Exploration strength for lower confidence bound.",
    )
    parser.add_argument(
        "--min-candidate-distance",
        type=float,
        default=0.04,
        help="Penalize candidates too close to previous evaluations.",
    )
    parser.add_argument("--no-baseline", action="store_true", help="Do not evaluate the conservative baseline.")
    parser.add_argument("--tasks", type=Path, help="Optional JSONL task file for multi-task evaluation.")
    parser.add_argument(
        "--compare-random",
        action="store_true",
        help="Also run a random-search baseline with the same evaluation budget.",
    )
    parser.add_argument("--json-output", type=Path, default=Path("token_optimization_result.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("token_optimization_history.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("token_optimization_report.md"))
    parser.add_argument("--html-output", type=Path, default=Path("token_optimization_report.html"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = TokenOptimizationConfig(
        iterations=args.iterations,
        initial_points=args.initial_points,
        quality_floor=args.quality_floor,
        quality_penalty=args.quality_penalty,
        random_state=args.random_state,
        evaluate_baseline=not args.no_baseline,
        acquisition=args.acquisition,
        exploration_weight=args.exploration_weight,
        min_candidate_distance=args.min_candidate_distance,
    )
    if args.tasks:
        objective = MultiTaskTokenQualityObjective(
            tasks=load_tasks_jsonl(args.tasks),
            quality_floor=policy.quality_floor,
            quality_penalty=policy.quality_penalty,
        )
    else:
        objective = TokenQualityObjective(
            quality_floor=policy.quality_floor,
            quality_penalty=policy.quality_penalty,
        )
    optimizer = TokenBayesOptimizer(
        search_space=SearchSpace.token_prompt_space(),
        objective=objective,
        config=policy,
    )
    result = optimizer.run()
    args.json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    result.write_csv(args.csv_output)
    result.write_markdown_report(args.report_output)
    result.write_html_report(args.html_output)
    print("Best configuration")
    print(f"  tokens: {result.best.tokens}")
    print(f"  quality: {result.best.quality:.4f}")
    print(f"  objective: {result.best.objective:.4f}")
    if result.baseline is not None:
        savings = result.savings_summary()
        print(f"  baseline tokens: {result.baseline.tokens}")
        print(f"  tokens saved: {savings['tokens_saved']} ({savings['token_reduction_percent']}%)")
    print(f"  config: {result.best.config}")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.report_output}")
    print(f"Wrote {args.html_output}")
    if args.compare_random:
        random_policy = TokenOptimizationConfig(
            iterations=args.iterations,
            initial_points=args.iterations,
            quality_floor=args.quality_floor,
            quality_penalty=args.quality_penalty,
            random_state=args.random_state,
            evaluate_baseline=not args.no_baseline,
            acquisition="random_search",
        )
        random_result = TokenBayesOptimizer(
            search_space=SearchSpace.token_prompt_space(),
            objective=objective,
            config=random_policy,
        ).run()
        random_json = args.json_output.with_name(args.json_output.stem + "_random" + args.json_output.suffix)
        random_csv = args.csv_output.with_name(args.csv_output.stem + "_random" + args.csv_output.suffix)
        random_report = args.report_output.with_name(args.report_output.stem + "_random" + args.report_output.suffix)
        random_html = args.html_output.with_name(args.html_output.stem + "_random" + args.html_output.suffix)
        random_json.write_text(random_result.to_json() + "\n", encoding="utf-8")
        random_result.write_csv(random_csv)
        random_result.write_markdown_report(random_report)
        random_result.write_html_report(random_html)
        token_delta = random_result.best.tokens - result.best.tokens
        objective_delta = random_result.best.objective - result.best.objective
        print("Random-search comparison")
        print(f"  random best tokens: {random_result.best.tokens}")
        print(f"  bayes token advantage: {token_delta}")
        print(f"  bayes objective advantage: {objective_delta:.4f}")
        print(f"Wrote {random_json}")
        print(f"Wrote {random_csv}")
        print(f"Wrote {random_report}")
        print(f"Wrote {random_html}")


if __name__ == "__main__":
    main()
