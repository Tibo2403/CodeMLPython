import csv
import pathlib
import tempfile

from ai_drug_discovery import (
    MoleculeExample,
    canonicalize_smiles,
    discover_candidates,
    drug_likeness_score,
    featurize_smiles,
    generate_candidates,
    load_examples,
    lipinski_violations,
    molecular_descriptors,
    rdkit_available,
    veber_violations,
    write_scores,
)


def test_featurize_smiles_returns_stable_numeric_vector():
    features = featurize_smiles("CC(=O)Oc1ccccc1C(=O)O")

    assert len(features) >= 13
    assert all(isinstance(value, float) for value in features)
    assert features[1] > 0
    assert features[10] > 0


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


def test_example_molecule_csv_runs_end_to_end():
    examples = load_examples(pathlib.Path("examples/molecules.csv"))

    scores = discover_candidates(examples, ["CCO", "c1ccccc1O"], top_n=3)

    assert len(examples) >= 3
    assert len(scores) == 3


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
