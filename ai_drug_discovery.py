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
import html
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
except ModuleNotFoundError:
    Chem = None
    Crippen = None
    Descriptors = None
    Lipinski = None
    rdFingerprintGenerator = None
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


@dataclass(frozen=True)
class ModelMetrics:
    """Basic regression metrics for the activity model."""

    mae: float
    r2: float
    n: int


@dataclass(frozen=True)
class RunMetadata:
    """Parameters and summary values for a drug discovery run."""

    n_examples: int
    n_seeds: int
    n_candidates: int
    n_ranked: int
    n_rejected: int
    top_n: int
    activity_weight: float
    drug_likeness_weight: float
    random_state: int
    rdkit_enabled: bool


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


def train_activity_model(examples: Sequence[MoleculeExample], random_state: int = 42) -> ActivityModel:
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
                    random_state=random_state,
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
    passes_filters = (
        score >= 0.55
        and hetero_atoms <= 12
        and molecular_weight <= 500.0
        and atom_count <= 55.0
        and complexity_proxy <= 18.0
    )
    return score, passes_filters


def molecular_descriptors(smiles: str) -> dict[str, Any]:
    """Return interpretable descriptors for reports and CSV exports."""

    rdkit_descriptors = _molecular_descriptors_with_rdkit(smiles)
    if rdkit_descriptors is not None:
        return rdkit_descriptors

    features = featurize_smiles(smiles)
    return {
        "molecular_weight": features[10],
        "logp": "",
        "tpsa": "",
        "h_donors": "",
        "h_acceptors": features[4],
        "rotatable_bonds": "",
        "rings": features[6] / 2.0,
        "heavy_atoms": features[2],
    }


def rank_candidates(
    model: ActivityModel,
    candidates: Sequence[str],
    top_n: int = 20,
    activity_weight: float = 0.75,
    drug_likeness_weight: float = 0.25,
) -> list[CandidateScore]:
    """Predict activity, apply filters, and return the best candidates."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    return score_candidates(
        model,
        candidates,
        activity_weight=activity_weight,
        drug_likeness_weight=drug_likeness_weight,
    )[:top_n]


def score_candidates(
    model: ActivityModel,
    candidates: Sequence[str],
    activity_weight: float = 0.75,
    drug_likeness_weight: float = 0.25,
) -> list[CandidateScore]:
    """Score every candidate without truncating the ranked output."""

    _validate_score_weights(activity_weight, drug_likeness_weight)
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
        final_score = float(prediction) * activity_weight + likeness * drug_likeness_weight
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

    return sorted(scored, key=lambda item: item.final_score, reverse=True)


def discover_candidates(
    labelled_molecules: Sequence[MoleculeExample],
    seed_smiles: Sequence[str],
    top_n: int = 20,
    activity_weight: float = 0.75,
    drug_likeness_weight: float = 0.25,
    random_state: int = 42,
) -> list[CandidateScore]:
    """End-to-end automated discovery method."""

    model = train_activity_model(labelled_molecules, random_state=random_state)
    candidates = generate_candidates(seed_smiles)
    return rank_candidates(
        model,
        candidates,
        top_n=top_n,
        activity_weight=activity_weight,
        drug_likeness_weight=drug_likeness_weight,
    )


def evaluate_activity_model(examples: Sequence[MoleculeExample], random_state: int = 42) -> ModelMetrics:
    """Evaluate the activity model with leave-one-out validation."""

    if len(examples) < 4:
        raise ValueError("At least four labelled molecules are required for evaluation.")

    actual: list[float] = []
    predicted: list[float] = []
    for index, held_out in enumerate(examples):
        training = [example for position, example in enumerate(examples) if position != index]
        model = train_activity_model(training, random_state=random_state)
        prediction = model.predict([featurize_smiles(held_out.smiles)])[0]
        actual.append(held_out.activity)
        predicted.append(float(prediction))

    mae = sum(abs(left - right) for left, right in zip(actual, predicted)) / len(actual)
    mean_actual = sum(actual) / len(actual)
    total_sum_squares = sum((value - mean_actual) ** 2 for value in actual)
    residual_sum_squares = sum((left - right) ** 2 for left, right in zip(actual, predicted))
    r2 = 0.0 if total_sum_squares == 0 else 1.0 - (residual_sum_squares / total_sum_squares)
    return ModelMetrics(mae=mae, r2=r2, n=len(examples))


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


def load_seed_smiles(path: Path) -> list[str]:
    """Load seed SMILES from a CSV with a smiles column."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "smiles" not in (reader.fieldnames or set()):
            raise ValueError("Seed CSV must contain a 'smiles' column.")
        return [row["smiles"] for row in reader if row.get("smiles")]


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
                "molecular_weight",
                "logp",
                "tpsa",
                "h_donors",
                "h_acceptors",
                "rotatable_bonds",
                "rings",
                "heavy_atoms",
            ],
        )
        writer.writeheader()
        for score in scores:
            descriptors = molecular_descriptors(score.smiles)
            writer.writerow(
                {
                    "smiles": score.smiles,
                    "predicted_activity": f"{score.predicted_activity:.4f}",
                    "drug_likeness_score": f"{score.drug_likeness_score:.4f}",
                    "final_score": f"{score.final_score:.4f}",
                    "passes_filters": score.passes_filters,
                    "lipinski_violations": score.lipinski_violations,
                    "veber_violations": score.veber_violations,
                    "molecular_weight": _format_descriptor(descriptors["molecular_weight"]),
                    "logp": _format_descriptor(descriptors["logp"]),
                    "tpsa": _format_descriptor(descriptors["tpsa"]),
                    "h_donors": _format_descriptor(descriptors["h_donors"]),
                    "h_acceptors": _format_descriptor(descriptors["h_acceptors"]),
                    "rotatable_bonds": _format_descriptor(descriptors["rotatable_bonds"]),
                    "rings": _format_descriptor(descriptors["rings"]),
                    "heavy_atoms": _format_descriptor(descriptors["heavy_atoms"]),
                }
            )


def write_rejections(path: Path, scores: Sequence[CandidateScore]) -> None:
    """Write candidates that failed prioritization filters to a CSV file."""

    rejected = [score for score in scores if not score.passes_filters]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "smiles",
                "reason",
                "predicted_activity",
                "drug_likeness_score",
                "final_score",
                "lipinski_violations",
                "veber_violations",
            ],
        )
        writer.writeheader()
        for score in rejected:
            writer.writerow(
                {
                    "smiles": score.smiles,
                    "reason": rejection_reason(score),
                    "predicted_activity": f"{score.predicted_activity:.4f}",
                    "drug_likeness_score": f"{score.drug_likeness_score:.4f}",
                    "final_score": f"{score.final_score:.4f}",
                    "lipinski_violations": score.lipinski_violations,
                    "veber_violations": score.veber_violations,
                }
            )


def rejection_reason(score: CandidateScore) -> str:
    """Explain why a candidate did not pass prioritization filters."""

    reasons: list[str] = []
    if score.drug_likeness_score < 0.55:
        reasons.append("drug_likeness_below_threshold")
    if score.lipinski_violations > 1:
        reasons.append("lipinski_violations")
    if score.veber_violations > 0:
        reasons.append("veber_violations")
    if not reasons:
        reasons.append("failed_filters")
    return ";".join(reasons)


def write_metrics(path: Path, metadata: RunMetadata, metrics: ModelMetrics | None) -> None:
    """Write run metadata and optional model metrics as JSON."""

    payload = {
        "n_examples": metadata.n_examples,
        "n_seeds": metadata.n_seeds,
        "n_candidates": metadata.n_candidates,
        "n_ranked": metadata.n_ranked,
        "n_rejected": metadata.n_rejected,
        "top_n": metadata.top_n,
        "activity_weight": metadata.activity_weight,
        "drug_likeness_weight": metadata.drug_likeness_weight,
        "random_state": metadata.random_state,
        "rdkit_enabled": metadata.rdkit_enabled,
        "model_metrics": None,
    }
    if metrics is not None:
        payload["model_metrics"] = {
            "n": metrics.n,
            "mae": metrics.mae,
            "r2": metrics.r2,
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_html_report(
    path: Path,
    ranked_scores: Sequence[CandidateScore],
    all_scores: Sequence[CandidateScore],
    metadata: RunMetadata,
    metrics: ModelMetrics | None,
) -> None:
    """Write a compact standalone HTML report for a discovery run."""

    rejected = [score for score in all_scores if not score.passes_filters]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AI Drug Discovery Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #f0f4f8; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d9e2ec; padding: 12px; background: #f8fafc; }}
    .muted {{ color: #627d98; }}
  </style>
</head>
<body>
  <h1>AI Drug Discovery Report</h1>
  <p class="muted">Educational triage report. Candidates require chemistry review and experimental validation.</p>
  { _html_summary(metadata, metrics) }
  <h2>Top Candidates</h2>
  { _html_score_table(ranked_scores, include_reason=False) }
  <h2>Rejected Candidates</h2>
  { _html_score_table(rejected, include_reason=True) }
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def _featurize_with_rdkit(smiles: str) -> list[float] | None:
    if (
        Chem is None
        or Descriptors is None
        or Crippen is None
        or Lipinski is None
        or rdFingerprintGenerator is None
        or rdMolDescriptors is None
    ):
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)
    fingerprint = generator.GetFingerprint(molecule)
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


def _molecular_descriptors_with_rdkit(smiles: str) -> dict[str, Any] | None:
    if Chem is None or Descriptors is None or Crippen is None or Lipinski is None or rdMolDescriptors is None:
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None

    return {
        "molecular_weight": Descriptors.MolWt(molecule),
        "logp": Crippen.MolLogP(molecule),
        "tpsa": rdMolDescriptors.CalcTPSA(molecule),
        "h_donors": Lipinski.NumHDonors(molecule),
        "h_acceptors": Lipinski.NumHAcceptors(molecule),
        "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
        "rings": rdMolDescriptors.CalcNumRings(molecule),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
    }


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


def _format_descriptor(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _html_summary(metadata: RunMetadata, metrics: ModelMetrics | None) -> str:
    metric_items = [
        ("Examples", metadata.n_examples),
        ("Seeds", metadata.n_seeds),
        ("Candidates", metadata.n_candidates),
        ("Ranked", metadata.n_ranked),
        ("Rejected", metadata.n_rejected),
        ("Top N", metadata.top_n),
        ("Activity weight", f"{metadata.activity_weight:.2f}"),
        ("Drug-likeness weight", f"{metadata.drug_likeness_weight:.2f}"),
        ("Random state", metadata.random_state),
        ("RDKit enabled", metadata.rdkit_enabled),
    ]
    if metrics is not None:
        metric_items.extend(
            [
                ("MAE", f"{metrics.mae:.4f}"),
                ("R2", f"{metrics.r2:.4f}"),
            ]
        )
    cards = "\n".join(
        f'<div class="metric"><strong>{html.escape(str(label))}</strong><br>{html.escape(str(value))}</div>'
        for label, value in metric_items
    )
    return f'<section class="summary">{cards}</section>'


def _html_score_table(scores: Sequence[CandidateScore], include_reason: bool) -> str:
    if not scores:
        return "<p>No candidates.</p>"
    headers = [
        "SMILES",
        "Predicted activity",
        "Drug-likeness",
        "Final score",
        "Passes filters",
        "Lipinski",
        "Veber",
    ]
    if include_reason:
        headers.append("Reason")
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    rows = []
    for score in scores:
        values: list[Any] = [
            score.smiles,
            f"{score.predicted_activity:.4f}",
            f"{score.drug_likeness_score:.4f}",
            f"{score.final_score:.4f}",
            score.passes_filters,
            score.lipinski_violations,
            score.veber_violations,
        ]
        if include_reason:
            values.append(rejection_reason(score))
        row_html = "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
        rows.append(f"<tr>{row_html}</tr>")
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _validate_score_weights(activity_weight: float, drug_likeness_weight: float) -> None:
    if activity_weight < 0 or drug_likeness_weight < 0:
        raise ValueError("Score weights must be non-negative.")
    if activity_weight == 0 and drug_likeness_weight == 0:
        raise ValueError("At least one score weight must be greater than zero.")
    total = activity_weight + drug_likeness_weight
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Score weights must sum to 1.0.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated AI-assisted molecule discovery from labelled SMILES data."
    )
    parser.add_argument("--training-csv", type=Path, required=True, help="CSV with smiles,activity columns.")
    parser.add_argument("--seed", nargs="+", default=[], help="Seed SMILES strings to mutate.")
    parser.add_argument("--seed-csv", type=Path, help="CSV with a smiles column containing seed molecules.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of candidates to keep.")
    parser.add_argument("--activity-weight", type=float, default=0.75, help="Final score weight for predicted activity.")
    parser.add_argument(
        "--drug-likeness-weight",
        type=float,
        default=0.25,
        help="Final score weight for drug-likeness.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Print leave-one-out model metrics.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state used by ML models.")
    parser.add_argument("--output", type=Path, default=Path("candidate_molecules.csv"), help="Output CSV path.")
    parser.add_argument("--rejected-output", type=Path, help="Optional CSV path for rejected candidates.")
    parser.add_argument("--metrics-output", type=Path, help="Optional JSON path for run metrics.")
    parser.add_argument("--report-output", type=Path, help="Optional HTML report path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    examples = load_examples(args.training_csv)
    seeds = list(args.seed)
    if args.seed_csv:
        seeds.extend(load_seed_smiles(args.seed_csv))
    if not seeds:
        raise SystemExit("Provide at least one seed through --seed or --seed-csv.")
    metrics = None
    if args.evaluate or args.metrics_output or args.report_output:
        metrics = evaluate_activity_model(examples, random_state=args.random_state)
    if args.evaluate and metrics is not None:
        print(f"Model metrics: n={metrics.n}, MAE={metrics.mae:.4f}, R2={metrics.r2:.4f}")
    model = train_activity_model(examples, random_state=args.random_state)
    candidates = generate_candidates(seeds)
    all_scores = score_candidates(
        model,
        candidates,
        activity_weight=args.activity_weight,
        drug_likeness_weight=args.drug_likeness_weight,
    )
    scores = all_scores[: args.top_n]
    metadata = RunMetadata(
        n_examples=len(examples),
        n_seeds=len(seeds),
        n_candidates=len(candidates),
        n_ranked=len(scores),
        n_rejected=sum(1 for score in all_scores if not score.passes_filters),
        top_n=args.top_n,
        activity_weight=args.activity_weight,
        drug_likeness_weight=args.drug_likeness_weight,
        random_state=args.random_state,
        rdkit_enabled=rdkit_available(),
    )
    write_scores(args.output, scores)
    print(f"Wrote {len(scores)} ranked candidates to {args.output}")
    if args.rejected_output:
        write_rejections(args.rejected_output, all_scores)
        print(f"Wrote {metadata.n_rejected} rejected candidates to {args.rejected_output}")
    if args.metrics_output:
        write_metrics(args.metrics_output, metadata, metrics)
        print(f"Wrote run metrics to {args.metrics_output}")
    if args.report_output:
        write_html_report(args.report_output, scores, all_scores, metadata, metrics)
        print(f"Wrote HTML report to {args.report_output}")


if __name__ == "__main__":
    main()
