import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from token_bayes_optimizer.optimizer import (
    GaussianProcessRegressor,
    MultiTaskTokenQualityObjective,
    SearchSpace,
    Task,
    TokenBayesOptimizer,
    TokenOptimizationConfig,
    TokenQualityObjective,
    acquisition_score,
    expected_improvement,
    load_tasks_jsonl,
    probability_improvement,
)


def test_default_objective_returns_tokens_quality_and_objective():
    objective = TokenQualityObjective(quality_floor=0.82)
    config = {
        "max_context_chars": 2400,
        "retrieval_top_k": 3,
        "summary_ratio": 0.45,
        "few_shot_examples": 1,
        "reasoning_level": "low",
        "format_style": "compact",
    }

    value, metadata = objective(config)

    assert value > 0
    assert metadata["tokens"] > 0
    assert 0.0 <= metadata["quality"] <= 1.0


def test_optimizer_finds_valid_best_result():
    optimizer = TokenBayesOptimizer(
        search_space=SearchSpace.token_prompt_space(),
        objective=TokenQualityObjective(),
        config=TokenOptimizationConfig(iterations=12, initial_points=4, candidate_pool=60, random_state=7),
    )

    result = optimizer.run()

    assert len(result.observations) == 12
    assert result.best.tokens > 0
    assert result.best.quality >= result.best.metadata["quality_floor"]
    assert result.baseline is not None
    assert "tokens_saved" in result.savings_summary()
    payload = json.loads(result.to_json())
    assert payload["optimization_config"]["acquisition"] == "expected_improvement"


def test_multitask_objective_returns_task_breakdown():
    objective = MultiTaskTokenQualityObjective(
        tasks=(
            Task("qa", "qa", "Answer a simple question."),
            Task("rag", "rag", "Answer with sources.", requires_retrieval=True, requires_reasoning=True),
        ),
        quality_floor=0.78,
    )
    config = {
        "max_context_chars": 3000,
        "retrieval_top_k": 4,
        "summary_ratio": 0.55,
        "few_shot_examples": 2,
        "reasoning_level": "low",
        "format_style": "balanced",
    }

    value, metadata = objective(config)

    assert value > 0
    assert metadata["tokens"] > 0
    assert len(metadata["tasks"]) == 2
    assert "worst_task_quality" in metadata


def test_load_tasks_jsonl_reads_example_file():
    tasks = load_tasks_jsonl(pathlib.Path("examples/tasks.jsonl"))

    assert len(tasks) >= 3
    assert any(task.requires_retrieval for task in tasks)


def test_result_can_write_csv():
    optimizer = TokenBayesOptimizer(
        search_space=SearchSpace.token_prompt_space(),
        objective=TokenQualityObjective(),
        config=TokenOptimizationConfig(iterations=6, initial_points=4, candidate_pool=20, random_state=8),
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

    assert "tokens" in text
    assert "config_json" in text
    assert "Token Optimization Report" in report_text
    assert "Savings" in report_text
    assert "<svg" in html_text


def test_pareto_front_contains_non_dominated_points():
    optimizer = TokenBayesOptimizer(
        search_space=SearchSpace.token_prompt_space(),
        objective=TokenQualityObjective(),
        config=TokenOptimizationConfig(iterations=10, initial_points=4, candidate_pool=30, random_state=9),
    )
    result = optimizer.run()

    front = result.pareto_front()

    assert front
    for candidate in front:
        assert not any(
            other.tokens <= candidate.tokens
            and other.quality >= candidate.quality
            and (other.tokens < candidate.tokens or other.quality > candidate.quality)
            for other in result.observations
        )


def test_expected_improvement_prefers_uncertainty_or_better_mean():
    assert expected_improvement((10.0, 1.0), best=8.0) >= 0.0
    assert expected_improvement((6.0, 0.5), best=8.0) > expected_improvement((10.0, 0.5), best=8.0)
    assert probability_improvement((6.0, 0.5), best=8.0) > probability_improvement((10.0, 0.5), best=8.0)
    assert acquisition_score((6.0, 0.5), best=8.0, strategy="lower_confidence_bound") > acquisition_score(
        (10.0, 0.5),
        best=8.0,
        strategy="lower_confidence_bound",
    )


def test_optimizer_supports_alternate_acquisition_strategy():
    optimizer = TokenBayesOptimizer(
        search_space=SearchSpace.token_prompt_space(),
        objective=TokenQualityObjective(quality_floor=0.78),
        config=TokenOptimizationConfig(
            iterations=10,
            initial_points=4,
            candidate_pool=40,
            random_state=11,
            acquisition="lower_confidence_bound",
            exploration_weight=1.8,
        ),
    )

    result = optimizer.run()

    assert len(result.observations) == 10
    assert result.best.tokens > 0


def test_optimizer_supports_random_search_strategy():
    optimizer = TokenBayesOptimizer(
        search_space=SearchSpace.token_prompt_space(),
        objective=TokenQualityObjective(quality_floor=0.78),
        config=TokenOptimizationConfig(
            iterations=8,
            initial_points=3,
            candidate_pool=10,
            random_state=12,
            acquisition="random_search",
        ),
    )

    result = optimizer.run()

    assert len(result.observations) == 8
    assert result.optimization_config.acquisition == "random_search"


def test_gaussian_process_predicts_finite_values():
    model = GaussianProcessRegressor.fit([[0.0], [1.0], [0.5]], [10.0, 20.0, 12.0])

    mean, std = model.predict([0.25])

    assert mean == mean
    assert std > 0
