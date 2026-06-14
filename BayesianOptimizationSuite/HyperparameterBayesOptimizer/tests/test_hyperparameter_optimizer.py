import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "BayesCore" / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bayes_core import Constraint, OptimizationConfig
from hyperparameter_bayes_optimizer.core import (
    HyperparameterBayesOptimizer,
    HyperparameterSearchSpace,
    SimulatedModelObjective,
)


def test_simulated_objective_returns_ml_metrics():
    objective = SimulatedModelObjective(min_f1=0.80)
    score, metadata = objective(
        {
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_estimators": 260,
            "regularization": 0.5,
            "subsample": 0.82,
            "model_family": "gradient_boosting",
            "decision_threshold": 0.48,
        }
    )

    assert score >= 0
    assert metadata["f1"] > 0.80
    assert metadata["latency_ms"] > 0


def test_optimizer_runs_with_constraints_and_noise_handling():
    optimizer = HyperparameterBayesOptimizer(
        search_space=HyperparameterSearchSpace.tabular_classification_space(),
        objective=SimulatedModelObjective(min_f1=0.78),
        config=OptimizationConfig(
            iterations=10,
            initial_points=4,
            candidate_pool=40,
            random_state=11,
            observation_repeats=2,
            gp_noise=0.001,
        ),
        constraints=(Constraint("f1", "min", 0.78),),
    )

    result = optimizer.run()

    assert len(result.observations) == 10
    assert result.best.f1 >= 0.78
    assert result.best.metadata["repeats"] == 2
    json.loads(result.to_json())


def test_reports_are_exported():
    optimizer = HyperparameterBayesOptimizer(
        search_space=HyperparameterSearchSpace.tabular_classification_space(),
        objective=SimulatedModelObjective(min_f1=0.78),
        config=OptimizationConfig(iterations=6, initial_points=3, candidate_pool=20, random_state=5),
        constraints=(Constraint("f1", "min", 0.78),),
    )
    result = optimizer.run()

    with tempfile.TemporaryDirectory() as directory:
        csv_path = pathlib.Path(directory) / "history.csv"
        html_path = pathlib.Path(directory) / "report.html"
        result.write_csv(csv_path)
        result.write_html_report(html_path)
        csv_text = csv_path.read_text(encoding="utf-8")
        html_text = html_path.read_text(encoding="utf-8")

    assert "f1" in csv_text
    assert "Hyperparameter Bayesian Optimization Report" in html_text
