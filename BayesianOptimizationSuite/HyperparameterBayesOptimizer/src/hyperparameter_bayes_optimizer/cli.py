"""CLI for hyperparameter Bayesian optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayes_core import Constraint, OptimizationConfig
from hyperparameter_bayes_optimizer.core import (
    HyperparameterBayesOptimizer,
    HyperparameterSearchSpace,
    SimulatedModelObjective,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize ML hyperparameters with BayesCore.")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--initial-points", type=int, default=8)
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--acquisition",
        choices=["expected_improvement", "probability_improvement", "lower_confidence_bound", "random_search"],
        default="expected_improvement",
    )
    parser.add_argument("--exploration-weight", type=float, default=1.25)
    parser.add_argument("--observation-repeats", type=int, default=1)
    parser.add_argument("--gp-noise", type=float, default=1e-6)
    parser.add_argument("--min-f1", type=float, default=0.82)
    parser.add_argument("--max-latency-ms", type=float, default=0.0, help="0 disables the latency constraint.")
    parser.add_argument("--json-output", type=Path, default=Path("hyperparameter_optimization_result.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("hyperparameter_optimization_history.csv"))
    parser.add_argument("--html-output", type=Path, default=Path("hyperparameter_optimization_report.html"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constraints = [Constraint("f1", "min", args.min_f1)]
    if args.max_latency_ms > 0:
        constraints.append(Constraint("latency_ms", "max", args.max_latency_ms))
    config = OptimizationConfig(
        iterations=args.iterations,
        initial_points=args.initial_points,
        candidate_pool=args.candidate_pool,
        random_state=args.random_state,
        acquisition=args.acquisition,
        exploration_weight=args.exploration_weight,
        observation_repeats=args.observation_repeats,
        gp_noise=args.gp_noise,
    )
    optimizer = HyperparameterBayesOptimizer(
        search_space=HyperparameterSearchSpace.tabular_classification_space(),
        objective=SimulatedModelObjective(min_f1=args.min_f1),
        config=config,
        constraints=tuple(constraints),
    )
    result = optimizer.run()
    args.json_output.write_text(result.to_json() + "\n", encoding="utf-8")
    result.write_csv(args.csv_output)
    result.write_html_report(args.html_output)
    print("Best hyperparameter configuration")
    print(f"  f1: {result.best.f1:.4f}")
    print(f"  accuracy: {result.best.accuracy:.4f}")
    print(f"  latency_ms: {result.best.latency_ms:.2f}")
    print(f"  training_cost: {result.best.training_cost:.4f}")
    print(f"  objective: {result.best.objective:.4f}")
    print(f"  config: {result.best.config}")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.html_output}")


if __name__ == "__main__":
    main()
