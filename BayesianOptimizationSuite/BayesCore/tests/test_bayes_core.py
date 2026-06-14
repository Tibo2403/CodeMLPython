import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bayes_core import (
    Constraint,
    GenericBayesianOptimizer,
    GenericOptimizationResult,
    OptimizationConfig,
    Parameter,
    SearchSpace,
    acquisition_score,
    expected_improvement,
    probability_improvement,
)


def bowl_objective(config):
    x = float(config["x"])
    value = (x - 0.25) ** 2
    return value, {"x": x, "quality": 1.0 - value, "cost": x}


def test_generic_optimizer_runs():
    space = SearchSpace((Parameter("x", "float", low=0.0, high=1.0),))
    optimizer = GenericBayesianOptimizer(
        search_space=space,
        objective=bowl_objective,
        config=OptimizationConfig(iterations=12, initial_points=4, candidate_pool=50, random_state=3),
    )

    result = optimizer.run()

    assert len(result.observations) == 12
    assert result.best.objective >= 0
    json.loads(result.to_json())


def test_result_exports_and_reload():
    space = SearchSpace((Parameter("x", "float", low=0.0, high=1.0),))
    result = GenericBayesianOptimizer(
        search_space=space,
        objective=bowl_objective,
        config=OptimizationConfig(iterations=6, initial_points=3, candidate_pool=20, random_state=7),
    ).run()

    with tempfile.TemporaryDirectory() as directory:
        json_path = pathlib.Path(directory) / "result.json"
        csv_path = pathlib.Path(directory) / "history.csv"
        result.write_json(json_path)
        result.write_csv(csv_path)
        loaded = GenericOptimizationResult.from_json(json_path.read_text(encoding="utf-8"))
        csv_text = csv_path.read_text(encoding="utf-8")

    assert loaded.best.objective == result.best.objective
    assert "quality" in csv_text


def test_constraints_choose_best_feasible_observation():
    space = SearchSpace((Parameter("x", "float", low=0.0, high=1.0),))
    optimizer = GenericBayesianOptimizer(
        search_space=space,
        objective=bowl_objective,
        config=OptimizationConfig(iterations=10, initial_points=4, candidate_pool=30, random_state=1),
        constraints=(Constraint("quality", "min", 0.90),),
    )

    result = optimizer.run()

    assert result.best.metadata["quality"] >= 0.90


def test_random_baseline_and_pareto_front():
    space = SearchSpace((Parameter("x", "float", low=0.0, high=1.0),))
    optimizer = GenericBayesianOptimizer(
        search_space=space,
        objective=bowl_objective,
        config=OptimizationConfig(iterations=8, initial_points=3, candidate_pool=25, random_state=9),
    )

    baseline = optimizer.run_random_baseline()
    front = baseline.pareto_front({"quality": "max", "cost": "min"})

    assert baseline.optimization_config.acquisition == "random_search"
    assert len(baseline.observations) == 8
    assert front


def test_observation_repeats_simplify_noise_handling():
    class NoisyObjective:
        def __init__(self):
            self.values = [1.0, 3.0, 5.0]
            self.index = 0

        def __call__(self, config):
            value = self.values[self.index % len(self.values)]
            self.index += 1
            return value, {"score": value, "label": "same"}

    space = SearchSpace((Parameter("x", "float", low=0.0, high=1.0),))
    result = GenericBayesianOptimizer(
        search_space=space,
        objective=NoisyObjective(),
        config=OptimizationConfig(
            iterations=1,
            initial_points=1,
            random_state=2,
            observation_repeats=3,
            gp_noise=0.01,
        ),
    ).run()

    observation = result.observations[0]

    assert observation.objective == 3.0
    assert observation.metadata["score"] == 3.0
    assert observation.metadata["score_std"] == 2.0
    assert observation.metadata["objective_std"] == 2.0
    assert observation.metadata["replicate_objectives"] == [1.0, 3.0, 5.0]
    assert observation.metadata["repeats"] == 3
    assert observation.metadata["label"] == "same"


def test_acquisition_functions_rank_better_candidates():
    assert expected_improvement((0.1, 0.2), best=0.2) > expected_improvement((0.5, 0.2), best=0.2)
    assert probability_improvement((0.1, 0.2), best=0.2) > probability_improvement((0.5, 0.2), best=0.2)
    assert acquisition_score((0.1, 0.2), best=0.2, strategy="lower_confidence_bound") > acquisition_score(
        (0.5, 0.2),
        best=0.2,
        strategy="lower_confidence_bound",
    )
