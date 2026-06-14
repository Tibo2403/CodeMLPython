# Creer un nouveau projet avec BayesCore

Ce guide decrit la marche a suivre pour deriver un nouveau projet metier depuis
le noyau `BayesCore`.

## 1. Nommer la decision

Commence par formuler la decision que l'outil doit aider a prendre.

Exemples:

- choisir une configuration de prompt moins couteuse;
- prioriser des molecules candidates;
- choisir une profondeur de scan securite;
- choisir une politique RH sous contraintes.

Une bonne decision doit etre actionnable. Si le resultat ne permet pas de faire
un choix concret, la fonction objectif sera floue.

## 2. Definir les parametres

Chaque parametre doit etre controlable par l'utilisateur ou par le systeme.

```python
from bayes_core import Parameter, SearchSpace


space = SearchSpace(
    parameters=(
        Parameter("temperature", "float", low=0.0, high=1.0),
        Parameter("top_k", "int", low=1, high=12),
        Parameter("profile", "categorical", choices=("fast", "balanced", "deep")),
    )
)
```

Evite les espaces trop larges au debut. Il vaut mieux optimiser 5 parametres
bien choisis que 30 parametres mal bornes.

## 3. Ecrire la fonction objectif

La fonction objectif recoit une configuration et retourne:

```python
score, metadata = objective(config)
```

Le score est minimise. Les metadonnees expliquent le score.

```python
def objective(config):
    cost = estimate_cost(config)
    quality = evaluate_quality(config)
    shortfall = max(0.0, 0.85 - quality)
    score = cost + 30000 * shortfall
    return score, {
        "cost": cost,
        "quality": quality,
        "quality_shortfall": shortfall,
    }
```

## 4. Lancer l'optimiseur

```python
from bayes_core import GenericBayesianOptimizer, OptimizationConfig


optimizer = GenericBayesianOptimizer(
    search_space=space,
    objective=objective,
    config=OptimizationConfig(
        iterations=30,
        initial_points=8,
        acquisition="expected_improvement",
        random_state=42,
    ),
)

result = optimizer.run()
print(result.best.config)
print(result.best.metadata)
```

## 5. Ajouter un adaptateur metier

Pour un projet professionnel, ne retourne pas seulement le resultat generique.
Cree un objet metier qui rend les metriques explicites:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectObservation:
    objective: float
    cost: float
    quality: float
    metadata: dict
```

Cela rend les rapports, tests et exports plus clairs.

## 6. Ajouter des exports

Minimum recommande:

- JSON complet pour rejouer le run;
- CSV pour analyse dans Excel ou pandas;
- rapport Markdown ou HTML pour lecture humaine;
- front de Pareto si le domaine a plusieurs compromis.

## 7. Tester

Ajoute au moins:

- un test de la fonction objectif;
- un test qui lance l'optimiseur avec peu d'iterations;
- un test d'export JSON ou CSV;
- un test de reproductibilite avec `random_state`.

## Exemple drug discovery

```python
def drug_objective(config):
    affinity = predict_binding_affinity(config)
    toxicity = predict_toxicity(config)
    admet_risk = predict_admet_risk(config)
    novelty = estimate_novelty(config)

    score = toxicity + admet_risk - 2.0 * affinity - 0.5 * novelty
    return score, {
        "affinity": affinity,
        "toxicity": toxicity,
        "admet_risk": admet_risk,
        "novelty": novelty,
    }
```

Le noyau ne valide pas qu'une molecule est rÃ©ellement utilisable. Il priorise
des candidats selon les metriques fournies. La validation scientifique reste
obligatoire.

## Exemple securite defensive

```python
def security_objective(config):
    result = run_local_static_scan(config)
    score = (
        result["elapsed_ms"] * 0.01
        - result["findings"] * 10
        - result["high_findings"] * 25
        - result["precision_proxy"] * 15
    )
    return score, result
```

Utilise ce type d'objectif uniquement sur du code local autorise.

