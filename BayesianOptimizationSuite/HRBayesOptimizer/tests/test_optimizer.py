import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hr_bayes_optimizer.optimizer import (
    GaussianProcessRegressor,
    HRBayesOptimizer,
    HROptimizationConfig,
    HRPolicyObjective,
    OptimizationResult,
    SearchSpace,
    expected_improvement,
    is_feasible,
    simulate_hr_metrics,
)
from hr_bayes_optimizer.core import (
    BayesianOptimizationConfig,
    BayesianOptimizer,
    Constraint,
    Parameter,
    SearchSpace as GenericSearchSpace,
    probability_slack_is_feasible,
)


def test_default_objective_returns_hr_metrics_and_objective():
    objective = HRPolicyObjective()
    config = {
        "sourcing_budget_eur": 3500,
        "interview_rounds": 3,
        "assessment_weight": 0.55,
        "onboarding_hours": 36,
        "remote_days_per_week": 3,
        "referral_bonus_eur": 1200,
        "screening_policy": "balanced",
    }

    value, metadata = objective(config)

    assert value > 0
    assert metadata["cost_eur"] > 0
    assert 0.0 <= metadata["quality"] <= 1.0
    assert 0.0 <= metadata["retention"] <= 1.0
    assert 0.0 <= metadata["fairness_gap"] <= 1.0


def test_simulation_is_deterministic():
    config = {
        "sourcing_budget_eur": 3500,
        "interview_rounds": 3,
        "assessment_weight": 0.55,
        "onboarding_hours": 36,
        "remote_days_per_week": 3,
        "referral_bonus_eur": 1200,
        "screening_policy": "balanced",
    }

    assert simulate_hr_metrics(config) == simulate_hr_metrics(config)


def test_optimizer_finds_valid_best_result():
    optimizer = HRBayesOptimizer(
        search_space=SearchSpace.hr_policy_space(),
        objective=HRPolicyObjective(),
        config=HROptimizationConfig(iterations=14, initial_points=5, candidate_pool=80, random_state=7),
    )

    result = optimizer.run()

    assert len(result.observations) == 14
    assert result.best.cost_eur > 0
    assert is_feasible(result.best)
    assert result.baseline is not None
    assert "cost_saved_eur" in result.savings_summary()
    json.loads(result.to_json())


def test_optimizer_averages_repeated_noisy_evaluations():
    optimizer = HRBayesOptimizer(
        search_space=SearchSpace.hr_policy_space(),
        objective=HRPolicyObjective(metric_noise=0.015, cost_noise_eur=125.0, random_state=11),
        config=HROptimizationConfig(
            iterations=10,
            initial_points=5,
            candidate_pool=40,
            observation_repeats=3,
            random_state=11,
        ),
    )

    result = optimizer.run()

    assert len(result.observations) == 10
    assert all(observation.metadata["repeats"] == 3 for observation in result.observations)
    assert any(observation.objective_std > 0 for observation in result.observations)
    assert "replicate_objectives" in result.observations[0].metadata


def test_result_can_write_reports():
    optimizer = HRBayesOptimizer(
        search_space=SearchSpace.hr_policy_space(),
        objective=HRPolicyObjective(),
        config=HROptimizationConfig(iterations=8, initial_points=5, candidate_pool=30, random_state=8),
    )
    result = optimizer.run()

    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "history.csv"
        report = pathlib.Path(directory) / "report.md"
        html_report = pathlib.Path(directory) / "report.html"
        result.write_csv(output)
        result.write_markdown_report(report)
        result.write_html_report(html_report)
        text = output.read_text(encoding="utf-8")
        report_text = report.read_text(encoding="utf-8")
        html_text = html_report.read_text(encoding="utf-8")

    assert "fairness_gap" in text
    assert "config_json" in text
    assert "HR Bayesian Optimization Report" in report_text
    assert "Best Policy" in report_text
    assert "<svg" in html_text


def test_result_can_resume_from_json_history():
    optimizer = HRBayesOptimizer(
        search_space=SearchSpace.hr_policy_space(),
        objective=HRPolicyObjective(),
        config=HROptimizationConfig(iterations=6, initial_points=4, candidate_pool=25, random_state=14),
    )
    first = optimizer.run()
    loaded = OptimizationResult.from_json(first.to_json())
    resumed_optimizer = HRBayesOptimizer(
        search_space=SearchSpace.hr_policy_space(),
        objective=HRPolicyObjective(),
        config=HROptimizationConfig(iterations=4, initial_points=4, candidate_pool=25, random_state=15),
    )

    resumed = resumed_optimizer.run(initial_observations=loaded.observations, baseline=loaded.baseline)

    assert len(resumed.observations) == 10
    assert resumed.observations[6].iteration == 6
    assert resumed.baseline == loaded.baseline
    json.loads(resumed.to_json())


def test_pareto_front_contains_non_dominated_points():
    optimizer = HRBayesOptimizer(
        search_space=SearchSpace.hr_policy_space(),
        objective=HRPolicyObjective(),
        config=HROptimizationConfig(iterations=12, initial_points=5, candidate_pool=40, random_state=9),
    )
    result = optimizer.run()

    front = result.pareto_front()

    assert front
    for candidate in front:
        assert not any(
            other.cost_eur <= candidate.cost_eur
            and other.quality >= candidate.quality
            and other.fairness_gap <= candidate.fairness_gap
            and (
                other.cost_eur < candidate.cost_eur
                or other.quality > candidate.quality
                or other.fairness_gap < candidate.fairness_gap
            )
            for other in result.observations
        )


def test_expected_improvement_prefers_uncertainty_or_better_mean():
    assert expected_improvement((10.0, 1.0), best=8.0) >= 0.0
    assert expected_improvement((6.0, 0.5), best=8.0) > expected_improvement((10.0, 0.5), best=8.0)


def test_gaussian_process_predicts_finite_values():
    model = GaussianProcessRegressor.fit(
        [[0.0], [1.0], [0.5]],
        [10.0, 20.0, 12.0],
        objective_noise=[0.1, 3.0, 0.2],
    )

    mean, std = model.predict([0.25])

    assert mean == mean
    assert std > 0
    assert model.length_scale in (0.25, 0.4, 0.65, 1.0, 1.6)


def test_core_optimizer_can_run_non_hr_objective():
    def objective(config):
        x = float(config["x"])
        y = str(config["mode"])
        mode_penalty = 0.0 if y == "fast" else 0.4
        value = (x - 0.35) ** 2 + mode_penalty
        return value, {"loss": value}

    optimizer = BayesianOptimizer(
        search_space=GenericSearchSpace(
            parameters=(
                Parameter("x", "float", low=0.0, high=1.0),
                Parameter("mode", "categorical", choices=("fast", "safe")),
            )
        ),
        objective=objective,
        config=BayesianOptimizationConfig(iterations=12, initial_points=4, candidate_pool=50, random_state=5),
    )

    result = optimizer.run()

    assert len(result.observations) == 12
    assert result.best.objective >= 0.0
    assert "loss" in result.best.metadata


def test_core_optimizer_supports_generic_constraints():
    def objective(config):
        x = float(config["x"])
        loss = (x - 0.8) ** 2
        score = x
        return loss, {"loss": loss, "score": score}

    optimizer = BayesianOptimizer(
        search_space=GenericSearchSpace(parameters=(Parameter("x", "float", low=0.0, high=1.0),)),
        objective=objective,
        config=BayesianOptimizationConfig(iterations=10, initial_points=4, candidate_pool=40, random_state=12),
        constraints=(Constraint("score", "min", 0.6),),
    )

    result = optimizer.run()

    assert len(result.observations) == 10
    assert result.best.metadata["score"] >= 0.6
    assert probability_slack_is_feasible((0.2, 0.01)) > probability_slack_is_feasible((-0.2, 0.01))
