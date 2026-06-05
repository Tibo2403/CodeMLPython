from ai_drug_discovery import (
    MoleculeExample,
    canonicalize_smiles,
    discover_candidates,
    drug_likeness_score,
    featurize_smiles,
    generate_candidates,
    lipinski_violations,
    rdkit_available,
    veber_violations,
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
