import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "BayesCore" / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from security_bayes_optimizer.core import BayesianOptimizer, OptimizationConfig, SearchSpace
from security_bayes_optimizer.scanner import SecurityScanObjective, scan_path, scan_text


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo_project"


def test_scan_text_detects_core_security_patterns():
    findings = scan_text(
        "API_KEY='abcdef1234567890abcdef'\neval(user)\nsubprocess.run(x, shell=True)\n",
        pathlib.Path("demo.py"),
        profile="balanced",
        severity_floor="low",
        entropy_threshold=3.0,
    )
    rule_ids = {finding.rule_id for finding in findings}

    assert "hardcoded_secret" in rule_ids
    assert "python_eval" in rule_ids
    assert "shell_true" in rule_ids


def test_scan_path_returns_objective_metadata():
    config = {
        "max_files": 100,
        "max_file_kb": 128,
        "entropy_threshold": 3.0,
        "include_tests": "yes",
        "rule_profile": "deep",
        "severity_floor": "low",
    }

    result = scan_path(DEMO, config)

    assert result["findings"] >= 4
    assert result["high_findings"] >= 2
    assert result["findings_list"]


def test_bayesian_optimizer_runs_security_objective():
    optimizer = BayesianOptimizer(
        search_space=SearchSpace.security_scan_space(),
        objective=SecurityScanObjective(DEMO),
        config=OptimizationConfig(iterations=8, initial_points=4, candidate_pool=30, random_state=4),
    )

    result = optimizer.run()

    assert len(result.observations) == 8
    assert result.best.findings >= 1
    assert json.loads(result.to_json())["best"]["findings"] == result.best.findings


def test_reports_are_exported():
    optimizer = BayesianOptimizer(
        search_space=SearchSpace.security_scan_space(),
        objective=SecurityScanObjective(DEMO),
        config=OptimizationConfig(iterations=6, initial_points=3, candidate_pool=20, random_state=5),
    )
    result = optimizer.run()

    with tempfile.TemporaryDirectory() as directory:
        csv_path = pathlib.Path(directory) / "history.csv"
        html_path = pathlib.Path(directory) / "report.html"
        result.write_csv(csv_path)
        result.write_html_report(html_path)
        csv_text = csv_path.read_text(encoding="utf-8")
        html_text = html_path.read_text(encoding="utf-8")

    assert "findings" in csv_text
    assert "Security Bayesian Optimization Report" in html_text
