import csv
import json
import pathlib
import tempfile

from codeml.drug_discovery import (
    MoleculeExample,
    RunMetadata,
    build_conformal_reliability_model,
    canonicalize_smiles,
    discover_candidates,
    drug_likeness_score,
    evaluate_activity_model,
    featurize_smiles,
    generate_candidates,
    lipinski_violations,
    load_examples,
    load_seed_smiles,
    molecular_descriptors,
    rank_candidates,
    rdkit_available,
    rejection_reason,
    reliability_assessment,
    score_candidates,
    train_activity_model,
    veber_violations,
    write_html_report,
    write_metrics,
    write_rejections,
    write_scores,
)


def test_featurize_smiles_returns_stable_numeric_vector():
    features = featurize_smiles("CC(=O)Oc1ccccc1C(=O)O")

    assert len(features) >= 13
    assert all(isinstance(value, float) for value in features)
    assert features[1] > 0
    assert molecular_descriptors("CC(=O)Oc1ccccc1C(=O)O")["molecular_weight"] > 0


def test_generate_candidates_adds_fragments_and_keeps_seed():
    candidates = generate_candidates(["CCO"], fragments=("F", "Cl"), max_candidates=10)

    assert "CCO" in candidates
    assert "CCOF" in candidates
    assert "CCOCl" in candidates


def test_drug_likeness_score_returns_bounded_score_and_boolean_filter():
    score, passes_filters = drug_likeness_score("CC(=O)Oc1ccccc1C(=O)O")

    assert 0.0 <= score <= 1.0
    assert isinstance(passes_filters, bool)


def test_medicinal_chemistry_filters_return_counts():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"

    assert lipinski_violations(smiles) >= 0
    assert veber_violations(smiles) >= 0


def test_molecular_descriptors_include_export_fields():
    descriptors = molecular_descriptors("CC(=O)Oc1ccccc1C(=O)O")

    assert "molecular_weight" in descriptors
    assert "h_acceptors" in descriptors
    assert descriptors["molecular_weight"] > 0


def test_canonicalize_smiles_uses_rdkit_when_available():
    canonical = canonicalize_smiles("CCO")

    assert canonical == "CCO"
    assert isinstance(rdkit_available(), bool)


def test_discover_candidates_ranks_generated_molecules():
    examples = [
        MoleculeExample("CCO", 0.20),
        MoleculeExample("CCN", 0.35),
        MoleculeExample("c1ccccc1O", 0.75),
        MoleculeExample("CC(=O)Oc1ccccc1C(=O)O", 0.90),
    ]

    scores = discover_candidates(examples, ["CCO", "c1ccccc1O"], top_n=5)

    assert len(scores) == 5
    assert scores == sorted(scores, key=lambda item: item.final_score, reverse=True)
    assert all(score.smiles for score in scores)
    assert all(score.lipinski_violations >= 0 for score in scores)
    assert all(score.veber_violations >= 0 for score in scores)


def test_score_weights_change_final_ranking_score():
    examples = [
        MoleculeExample("CCO", 0.20),
        MoleculeExample("CCN", 0.35),
        MoleculeExample("c1ccccc1O", 0.75),
    ]
    model = train_activity_model(examples)

    activity_first = rank_candidates(model, ["CCO"], top_n=1, activity_weight=1.0, drug_likeness_weight=0.0)[0]
    likeness_first = rank_candidates(model, ["CCO"], top_n=1, activity_weight=0.0, drug_likeness_weight=1.0)[0]

    assert activity_first.final_score == activity_first.predicted_activity
    assert likeness_first.final_score == likeness_first.drug_likeness_score


def test_score_weights_must_sum_to_one():
    examples = [
        MoleculeExample("CCO", 0.20),
        MoleculeExample("CCN", 0.35),
        MoleculeExample("c1ccccc1O", 0.75),
    ]
    model = train_activity_model(examples)

    try:
        rank_candidates(model, ["CCO"], activity_weight=0.8, drug_likeness_weight=0.8)
    except ValueError as error:
        assert "sum to 1.0" in str(error)
    else:
        raise AssertionError("Expected invalid score weights to raise ValueError.")


def test_example_molecule_csv_runs_end_to_end():
    examples = load_examples(pathlib.Path("examples/molecules.csv"))

    scores = discover_candidates(examples, ["CCO", "c1ccccc1O"], top_n=3)

    assert len(examples) >= 3
    assert len(scores) == 3


def test_conformal_reliability_model_scores_candidate_uncertainty():
    examples = load_examples(pathlib.Path("examples/molecules.csv"))
    model = train_activity_model(examples, random_state=123)
    reliability_model = build_conformal_reliability_model(examples, confidence=0.9, random_state=123)

    scores = score_candidates(model, ["CCO"], reliability_model=reliability_model)

    assert reliability_model.confidence == 0.9
    assert reliability_model.residual_quantile >= 0
    assert scores[0].prediction_lower is not None
    assert scores[0].prediction_upper is not None
    assert scores[0].prediction_lower <= scores[0].predicted_activity <= scores[0].prediction_upper
    assert scores[0].uncertainty_width is not None
    assert scores[0].applicability_label in {"in_domain", "near_domain", "out_of_domain"}
    assert scores[0].reliability_label in {"high", "medium", "low"}
    assert scores[0].decision in {"prioritize", "review", "deprioritize"}


def test_reliability_assessment_can_be_disabled():
    assessment = reliability_assessment(
        predicted_activity=0.7,
        features=featurize_smiles("CCO"),
        passes_filters=True,
        reliability_model=None,
    )

    assert assessment["prediction_lower"] is None
    assert assessment["applicability_label"] == "not_evaluated"
    assert assessment["decision"] == "review"


def test_seed_csv_loads_smiles_column():
    seeds = load_seed_smiles(pathlib.Path("examples/seeds.csv"))

    assert seeds == ["CCO", "c1ccccc1O", "CCN"]


def test_evaluate_activity_model_returns_metrics():
    examples = load_examples(pathlib.Path("examples/molecules.csv"))

    metrics = evaluate_activity_model(examples)

    assert metrics.n == len(examples)
    assert metrics.mae >= 0
    assert isinstance(metrics.r2, float)


def test_random_state_keeps_model_evaluation_reproducible():
    examples = load_examples(pathlib.Path("examples/molecules.csv"))

    first = evaluate_activity_model(examples, random_state=123)
    second = evaluate_activity_model(examples, random_state=123)

    assert first == second


def test_write_scores_exports_descriptors():
    examples = [
        MoleculeExample("CCO", 0.20),
        MoleculeExample("CCN", 0.35),
        MoleculeExample("c1ccccc1O", 0.75),
    ]
    scores = discover_candidates(examples, ["CCO"], top_n=2)
    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "candidates.csv"
        write_scores(output, scores)
        rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert rows
    assert "molecular_weight" in rows[0]
    assert "tpsa" in rows[0]
    assert rows[0]["molecular_weight"]
    assert "prediction_lower" in rows[0]
    assert "applicability_label" in rows[0]
    assert "decision" in rows[0]


def test_score_candidates_returns_full_ranked_list():
    examples = load_examples(pathlib.Path("examples/molecules.csv"))
    model = train_activity_model(examples)
    candidates = generate_candidates(["CCO", "c1ccccc1O"])

    scores = score_candidates(model, candidates)

    assert len(scores) == len(candidates)
    assert scores == sorted(scores, key=lambda item: item.final_score, reverse=True)


def test_write_rejections_exports_filter_failures():
    examples = load_examples(pathlib.Path("examples/molecules.csv"))
    model = train_activity_model(examples)
    candidates = ["C" * 80, "CCO"]
    scores = score_candidates(model, candidates)

    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "rejected.csv"
        write_rejections(output, scores)
        rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert rows
    assert "reason" in rows[0]
    assert rows[0]["smiles"]
    assert "reliability_label" in rows[0]


def test_rejection_reason_names_failed_filters():
    examples = load_examples(pathlib.Path("examples/molecules.csv"))
    model = train_activity_model(examples)
    rejected = score_candidates(model, ["C" * 80])[-1]

    reason = rejection_reason(rejected)

    assert reason


def test_write_metrics_exports_json_payload():
    metadata = RunMetadata(
        n_examples=10,
        n_seeds=3,
        n_candidates=20,
        n_ranked=5,
        n_rejected=2,
        top_n=5,
        activity_weight=0.7,
        drug_likeness_weight=0.3,
        random_state=123,
        rdkit_enabled=rdkit_available(),
        conformal_enabled=True,
        conformal_confidence=0.9,
        conformal_residual_quantile=0.12,
        applicability_threshold=1.5,
    )
    metrics = evaluate_activity_model(load_examples(pathlib.Path("examples/molecules.csv")), random_state=123)

    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "metrics.json"
        write_metrics(output, metadata, metrics)
        payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["random_state"] == 123
    assert payload["model_metrics"]["n"] == metrics.n
    assert payload["activity_weight"] == 0.7
    assert payload["conformal_enabled"] is True
    assert payload["conformal_confidence"] == 0.9


def test_write_html_report_creates_readable_report():
    examples = load_examples(pathlib.Path("examples/molecules.csv"))
    model = train_activity_model(examples, random_state=123)
    all_scores = score_candidates(model, generate_candidates(["CCO"]), activity_weight=0.7, drug_likeness_weight=0.3)
    ranked = all_scores[:3]
    metadata = RunMetadata(
        n_examples=len(examples),
        n_seeds=1,
        n_candidates=len(all_scores),
        n_ranked=len(ranked),
        n_rejected=sum(1 for score in all_scores if not score.passes_filters),
        top_n=3,
        activity_weight=0.7,
        drug_likeness_weight=0.3,
        random_state=123,
        rdkit_enabled=rdkit_available(),
    )

    with tempfile.TemporaryDirectory() as directory:
        output = pathlib.Path(directory) / "report.html"
        write_html_report(output, ranked, all_scores, metadata, metrics=None)
        report = output.read_text(encoding="utf-8")

    assert "AI Drug Discovery Report" in report
    assert "Top Candidates" in report
    assert "Random state" in report
    assert "Conformal reliability" in report
    assert "Decision" in report
