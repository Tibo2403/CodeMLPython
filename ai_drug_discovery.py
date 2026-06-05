"""AI-assisted candidate molecule discovery workflow.

This module is a lightweight, automatable template for early-stage drug
discovery experiments. It uses SMILES strings, simple molecular descriptors,
candidate generation, a supervised scoring model, and drug-likeness filters.

It is intended for education and research prototyping only. Promising
candidates still require chemistry review, synthesis feasibility analysis,
toxicity testing, and experimental validation.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    np = None
    RandomForestRegressor = None
    Pipeline = None
    StandardScaler = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
except ModuleNotFoundError:
    Chem = None
    AllChem = None
    Crippen = None
    Descriptors = None
    Lipinski = None
    rdMolDescriptors = None


ATOM_PATTERN = re.compile(r"Cl|Br|[BCNOFPSIbcno]")
DEFAULT_FRAGMENTS = ("F", "Cl", "Br", "C", "N", "O", "C(=O)N", "OC", "CN")
MORGAN_BITS = 128


@dataclass(frozen=True)
class MoleculeExample:
    """A labelled molecule used to train the activity scoring model."""

    smiles: str
    activity: float


@dataclass(frozen=True)
class CandidateScore:
    """Ranked output for a generated candidate molecule."""

    smiles: str
    predicted_activity: float
    drug_likeness_score: float
    final_score: float
    passes_filters: bool
    lipinski_violations: int = 0
    veber_violations: int = 0


class ActivityModel(Protocol):
    """Minimal model interface used by the ranking workflow."""

    def predict(self, features: Sequence[Sequence[float]]) -> Sequence[float]:
        """Predict activity scores for feature vectors."""


@dataclass
class SimilarityActivityModel:
    """Dependency-free fallback model based on nearest descriptor similarity."""

    training_features: list[list[float]]
    activities: list[float]

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        predictions: list[float] = []
        for feature in features:
            weights: list[float] = []
            weighted_scores: list[float] = []
            for known_feature, activity in zip(self.training_features, self.activities):
                distance = _euclidean_distance(feature, known_feature)
                weight = 1.0 / (1.0 + distance)
                weights.append(weight)
                weighted_scores.append(weight * activity)
            predictions.append(sum(weighted_scores) / max(sum(weights), 1e-9))
        return predictions


def featurize_smiles(smiles: str) -> list[float]:
    """Convert a SMILES string into simple numeric descriptors.

    When RDKit is installed, the vector combines medicinal chemistry
    descriptors and a Morgan fingerprint. Otherwise, the function falls back to
    a dependency-light descriptor set so the repository remains runnable in CI.
    """

    rdkit_features = _featurize_with_rdkit(smiles)
    if rdkit_features is not None:
        return rdkit_features

    atoms = ATOM_PATTERN.findall(smiles)
    atom_count = len(atoms)
    heavy_atom_count = sum(1 for atom in atoms if atom not in {"H"})
    aromatic_atoms = sum(1 for atom in atoms if atom.islower())
    hetero_atoms = sum(1 for atom in atoms if atom in {"N", "O", "S", "P", "F", "Cl", "Br", "I"})
    carbon_atoms = sum(1 for atom in atoms if atom in {"C", "c"})
    ring_markers = sum(char.isdigit() for char in smiles)
    branches = smiles.count("(")
    double_bonds = smiles.count("=")
    triple_bonds = smiles.count("#")
    approximate_weight = _approximate_molecular_weight(atoms)
    polarity_proxy = hetero_atoms / max(atom_count, 1)
    complexity_proxy = branches + ring_markers + double_bonds + triple_bonds

    return [
        float(len(smiles)),
        float(atom_count),
        float(heavy_atom_count),
        float(carbon_atoms),
        float(hetero_atoms),
        float(aromatic_atoms),
        float(ring_markers),
        float(branches),
        float(double_bonds),
        float(triple_bonds),
        approximate_weight,
        polarity_proxy,
        float(complexity_proxy),
    ]


def train_activity_model(examples: Sequence[MoleculeExample]) -> ActivityModel:
    """Train a QSAR-like model that predicts activity from SMILES features."""

    if len(examples) < 3:
        raise ValueError("At least three labelled molecules are required.")

    x_train = [featurize_smiles(example.smiles) for example in examples]
    y_train = [example.activity for example in examples]

    if np is None or Pipeline is None or StandardScaler is None or RandomForestRegressor is None:
        return SimilarityActivityModel(training_features=x_train, activities=y_train)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    random_state=42,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    model.fit(np.array(x_train, dtype=float), np.array(y_train, dtype=float))
    return model


def generate_candidates(
    seed_smiles: Iterable[str],
    fragments: Sequence[str] = DEFAULT_FRAGMENTS,
    max_candidates: int = 250,
) -> list[str]:
    """Generate new candidate SMILES by small substituent additions.

    This intentionally stays conservative: it appends or substitutes small
    fragments so the output is useful for automated ranking demonstrations.
    """

    candidates: set[str] = set()
    for smiles in seed_smiles:
        normalized = canonicalize_smiles(smiles.strip())
        if not normalized:
            continue
        candidates.add(normalized)

        for fragment in fragments:
            _add_candidate(candidates, f"{normalized}{fragment}")
            if normalized.endswith("C"):
                _add_candidate(candidates, f"{normalized[:-1]}{fragment}")
            if "c1ccccc1" in normalized:
                _add_candidate(candidates, normalized.replace("c1ccccc1", f"c1ccc({fragment})cc1", 1))

            if len(candidates) >= max_candidates:
                return sorted(candidates)

    return sorted(candidates)


def drug_likeness_score(smiles: str) -> tuple[float, bool]:
    """Estimate whether a molecule is worth prioritizing.

    The score loosely imitates early filters such as molecular weight and
    polarity bounds. It is a triage heuristic, not a medicinal chemistry rule.
    """

    rdkit_score = _rdkit_drug_likeness_score(smiles)
    if rdkit_score is not None:
        return rdkit_score

    features = featurize_smiles(smiles)
    atom_count = features[1]
    hetero_atoms = features[4]
    molecular_weight = features[10]
    polarity_proxy = features[11]
    complexity_proxy = features[12]

    penalties = 0.0
    penalties += _distance_penalty(molecular_weight, low=120.0, high=500.0, scale=120.0)
    penalties += _distance_penalty(atom_count, low=8.0, high=55.0, scale=25.0)
    penalties += _distance_penalty(polarity_proxy, low=0.05, high=0.45, scale=0.35)
    penalties += _distance_penalty(complexity_proxy, low=0.0, high=18.0, scale=12.0)

    score = max(0.0, min(1.0, 1.0 - penalties))
    passes_filters = score >= 0.55 and hetero_atoms <= 12
    return score, passes_filters


def rank_candidates(
    model: ActivityModel,
    candidates: Sequence[str],
    top_n: int = 20,
) -> list[CandidateScore]:
    """Predict activity, apply filters, and return the best candidates."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if not candidates:
        return []

    feature_matrix = [featurize_smiles(smiles) for smiles in candidates]
    if np is not None and not isinstance(model, SimilarityActivityModel):
        feature_matrix = np.array(feature_matrix, dtype=float)
    predictions = model.predict(feature_matrix)
    scored: list[CandidateScore] = []

    for smiles, prediction in zip(candidates, predictions):
        likeness, passes_filters = drug_likeness_score(smiles)
        lipinski = lipinski_violations(smiles)
        veber = veber_violations(smiles)
        final_score = float(prediction) * 0.75 + likeness * 0.25
        scored.append(
            CandidateScore(
                smiles=smiles,
                predicted_activity=float(prediction),
                drug_likeness_score=likeness,
                final_score=final_score,
                passes_filters=passes_filters,
                lipinski_violations=lipinski,
                veber_violations=veber,
            )
        )

    return sorted(scored, key=lambda item: item.final_score, reverse=True)[:top_n]


def discover_candidates(
    labelled_molecules: Sequence[MoleculeExample],
    seed_smiles: Sequence[str],
    top_n: int = 20,
) -> list[CandidateScore]:
    """End-to-end automated discovery method."""

    model = train_activity_model(labelled_molecules)
    candidates = generate_candidates(seed_smiles)
    return rank_candidates(model, candidates, top_n=top_n)


def rdkit_available() -> bool:
    """Return whether the optional RDKit chemistry backend is available."""

    return Chem is not None


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize a SMILES string with RDKit when possible."""

    if not smiles:
        return ""
    if Chem is None:
        return smiles
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True)


def lipinski_violations(smiles: str) -> int:
    """Count Lipinski rule-of-five violations.

    Returns a heuristic fallback count when RDKit is unavailable.
    """

    if Chem is None or Descriptors is None or Crippen is None or Lipinski is None:
        features = featurize_smiles(smiles)
        molecular_weight = features[10]
        hetero_atoms = features[4]
        polarity_proxy = features[11]
        return sum(
            [
                molecular_weight > 500.0,
                hetero_atoms > 10.0,
                polarity_proxy > 0.45,
            ]
        )

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return 4

    return sum(
        [
            Descriptors.MolWt(molecule) > 500.0,
            Crippen.MolLogP(molecule) > 5.0,
            Lipinski.NumHDonors(molecule) > 5,
            Lipinski.NumHAcceptors(molecule) > 10,
        ]
    )


def veber_violations(smiles: str) -> int:
    """Count Veber oral bioavailability filter violations."""

    if Chem is None or Descriptors is None or rdMolDescriptors is None:
        features = featurize_smiles(smiles)
        complexity_proxy = features[12]
        polarity_proxy = features[11]
        return sum([complexity_proxy > 18.0, polarity_proxy > 0.45])

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return 2

    return sum(
        [
            Lipinski.NumRotatableBonds(molecule) > 10,
            rdMolDescriptors.CalcTPSA(molecule) > 140.0,
        ]
    )


def load_examples(path: Path) -> list[MoleculeExample]:
    """Load labelled molecules from a CSV with smiles and activity columns."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"smiles", "activity"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("CSV must contain 'smiles' and 'activity' columns.")
        return [MoleculeExample(row["smiles"], float(row["activity"])) for row in reader]


def write_scores(path: Path, scores: Sequence[CandidateScore]) -> None:
    """Write ranked candidates to a CSV file."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "smiles",
                "predicted_activity",
                "drug_likeness_score",
                "final_score",
                "passes_filters",
                "lipinski_violations",
                "veber_violations",
            ],
        )
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "smiles": score.smiles,
                    "predicted_activity": f"{score.predicted_activity:.4f}",
                    "drug_likeness_score": f"{score.drug_likeness_score:.4f}",
                    "final_score": f"{score.final_score:.4f}",
                    "passes_filters": score.passes_filters,
                    "lipinski_violations": score.lipinski_violations,
                    "veber_violations": score.veber_violations,
                }
            )


def _featurize_with_rdkit(smiles: str) -> list[float] | None:
    if (
        Chem is None
        or AllChem is None
        or Descriptors is None
        or Crippen is None
        or Lipinski is None
        or rdMolDescriptors is None
    ):
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=MORGAN_BITS)
    fingerprint_bits = [float(bit) for bit in fingerprint.ToBitString()]
    descriptors = [
        float(Descriptors.MolWt(molecule)),
        float(Crippen.MolLogP(molecule)),
        float(rdMolDescriptors.CalcTPSA(molecule)),
        float(Lipinski.NumHDonors(molecule)),
        float(Lipinski.NumHAcceptors(molecule)),
        float(Lipinski.NumRotatableBonds(molecule)),
        float(rdMolDescriptors.CalcNumRings(molecule)),
        float(molecule.GetNumHeavyAtoms()),
        float(lipinski_violations(smiles)),
        float(veber_violations(smiles)),
    ]
    return descriptors + fingerprint_bits


def _rdkit_drug_likeness_score(smiles: str) -> tuple[float, bool] | None:
    if Chem is None or Descriptors is None or Crippen is None or Lipinski is None or rdMolDescriptors is None:
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return 0.0, False

    molecular_weight = Descriptors.MolWt(molecule)
    logp = Crippen.MolLogP(molecule)
    tpsa = rdMolDescriptors.CalcTPSA(molecule)
    donors = Lipinski.NumHDonors(molecule)
    acceptors = Lipinski.NumHAcceptors(molecule)
    rotatable_bonds = Lipinski.NumRotatableBonds(molecule)

    penalties = 0.0
    penalties += _distance_penalty(molecular_weight, low=120.0, high=500.0, scale=120.0)
    penalties += _distance_penalty(logp, low=-1.0, high=5.0, scale=4.0)
    penalties += _distance_penalty(tpsa, low=20.0, high=140.0, scale=80.0)
    penalties += _distance_penalty(float(donors), low=0.0, high=5.0, scale=8.0)
    penalties += _distance_penalty(float(acceptors), low=0.0, high=10.0, scale=10.0)
    penalties += _distance_penalty(float(rotatable_bonds), low=0.0, high=10.0, scale=10.0)

    lipinski = lipinski_violations(smiles)
    veber = veber_violations(smiles)
    score = max(0.0, min(1.0, 1.0 - penalties - (lipinski * 0.08) - (veber * 0.08)))
    return score, lipinski <= 1 and veber == 0 and score >= 0.55


def _add_candidate(candidates: set[str], smiles: str) -> None:
    canonical = canonicalize_smiles(smiles)
    if canonical:
        candidates.add(canonical)


def _approximate_molecular_weight(atoms: Sequence[str]) -> float:
    weights = {
        "B": 10.81,
        "C": 12.01,
        "c": 12.01,
        "N": 14.01,
        "n": 14.01,
        "O": 16.00,
        "o": 16.00,
        "F": 19.00,
        "P": 30.97,
        "S": 32.07,
        "s": 32.07,
        "Cl": 35.45,
        "Br": 79.90,
        "I": 126.90,
    }
    return float(sum(weights.get(atom, 0.0) for atom in atoms))


def _distance_penalty(value: float, low: float, high: float, scale: float) -> float:
    if low <= value <= high:
        return 0.0
    distance = low - value if value < low else value - high
    return min(1.0, math.log1p(distance) / scale)


def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated AI-assisted molecule discovery from labelled SMILES data."
    )
    parser.add_argument("--training-csv", type=Path, required=True, help="CSV with smiles,activity columns.")
    parser.add_argument("--seed", nargs="+", required=True, help="Seed SMILES strings to mutate.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of candidates to keep.")
    parser.add_argument("--output", type=Path, default=Path("candidate_molecules.csv"), help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    examples = load_examples(args.training_csv)
    scores = discover_candidates(examples, args.seed, top_n=args.top_n)
    write_scores(args.output, scores)
    print(f"Wrote {len(scores)} ranked candidates to {args.output}")


if __name__ == "__main__":
    main()
