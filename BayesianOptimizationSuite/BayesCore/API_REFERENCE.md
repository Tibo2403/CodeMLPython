# Reference API BayesCore

Cette reference resume les classes et fonctions principales du noyau.

## Types

```python
Config = dict[str, float | str]
Objective = Callable[[Config], tuple[float, dict[str, Any]]]
```

Une `Config` contient les valeurs d'une configuration candidate.
Une `Objective` retourne un score a minimiser et des metadonnees.

## Constraint

```python
Constraint(metric: str, operator: str, threshold: float)
```

Contrainte legere evaluee depuis les metadonnees d'une observation.

| Operateur | Signification |
| --- | --- |
| `min` | La metrique doit etre superieure ou egale au seuil |
| `max` | La metrique doit etre inferieure ou egale au seuil |

Exemple:

```python
Constraint("quality", "min", 0.85)
Constraint("latency_ms", "max", 250)
```

Si au moins une observation satisfait les contraintes, le meilleur resultat est
choisi parmi les observations faisables. Sinon, le noyau retourne le meilleur
score global.

## Parameter

```python
Parameter(
    name: str,
    kind: str,
    low: float | None = None,
    high: float | None = None,
    choices: tuple[str, ...] = (),
)
```

Parametre optimisable.

| Champ | Role |
| --- | --- |
| `name` | Nom de la cle dans la configuration |
| `kind` | `float`, `int` ou `categorical` |
| `low` / `high` | Bornes numeriques |
| `choices` | Valeurs possibles pour un categoriel |

Exemples:

```python
Parameter("temperature", "float", low=0.0, high=1.0)
Parameter("top_k", "int", low=1, high=10)
Parameter("profile", "categorical", choices=("fast", "balanced", "deep"))
```

## SearchSpace

```python
SearchSpace(parameters: tuple[Parameter, ...])
```

Regroupe les parametres.

Methodes utiles:

| Methode | Role |
| --- | --- |
| `sample(rng)` | Genere une configuration aleatoire |
| `encode(config)` | Transforme une configuration en vecteur numerique |

## OptimizationConfig

```python
OptimizationConfig(
    iterations=30,
    initial_points=8,
    candidate_pool=300,
    random_state=42,
    acquisition="expected_improvement",
    exploration_weight=1.25,
    min_candidate_distance=0.04,
    observation_repeats=1,
    gp_noise=1e-6,
)
```

| Champ | Role |
| --- | --- |
| `iterations` | Nombre total d'evaluations |
| `initial_points` | Evaluations aleatoires avant le modele |
| `candidate_pool` | Nombre de candidats compares a chaque iteration |
| `random_state` | Seed de reproductibilite |
| `acquisition` | Strategie de choix du prochain candidat |
| `exploration_weight` | Force d'exploration pour `lower_confidence_bound` |
| `min_candidate_distance` | Penalise les candidats trop proches |
| `observation_repeats` | Nombre de repetitions par configuration |
| `gp_noise` | Bruit minimal ajoute au Gaussian Process |

Pour une fonction objectif bruitee, commence simplement par:

```python
OptimizationConfig(observation_repeats=3, gp_noise=0.01)
```

Le noyau moyenne les repetitions et ajoute `objective_std`,
`replicate_objectives`, `repeats` et les champs `<metric>_std` quand les
metadonnees sont numeriques.

## GenericBayesianOptimizer

```python
GenericBayesianOptimizer(
    search_space: SearchSpace,
    objective: Objective,
    config: OptimizationConfig = OptimizationConfig(),
    constraints: tuple[Constraint, ...] = (),
)
```

Usage:

```python
result = GenericBayesianOptimizer(space, objective).run()
print(result.best.config)
print(result.best.objective)
print(result.best.metadata)
```

Methodes:

| Methode | Role |
| --- | --- |
| `run()` | Lance l'optimisation bayesienne |
| `run(initial_observations=...)` | Reprend depuis des observations existantes |
| `run_random_baseline()` | Lance une baseline aleatoire au meme budget |

## GenericObservation

Observation produite apres chaque evaluation.

| Champ | Role |
| --- | --- |
| `iteration` | Index de l'evaluation |
| `config` | Configuration testee |
| `objective` | Score obtenu |
| `metadata` | Informations metier |

## GenericOptimizationResult

Resultat final.

| Champ | Role |
| --- | --- |
| `best` | Meilleure observation |
| `observations` | Historique complet |
| `optimization_config` | Configuration de l'optimiseur |

Methode:

```python
json_text = result.to_json()
result.write_json("result.json")
result.write_csv("history.csv")
front = result.pareto_front({"quality": "max", "cost": "min"})
loaded = GenericOptimizationResult.from_json(json_text)
```

## Fonctions d'acquisition

| Fonction | Usage |
| --- | --- |
| `expected_improvement(prediction, best)` | Amelioration attendue |
| `probability_improvement(prediction, best)` | Probabilite de battre le meilleur score |
| `lower_confidence_bound(prediction, exploration_weight)` | Exploration par incertitude |
| `acquisition_score(...)` | Routeur selon le nom de strategie |

## Erreurs frequentes

| Probleme | Cause probable |
| --- | --- |
| `Unsupported parameter kind` | `kind` n'est pas `float`, `int` ou `categorical` |
| `needs low/high bounds` | Parametre numerique sans bornes |
| `needs choices` | Parametre categoriel sans valeurs |
| `Unknown acquisition strategy` | Nom d'acquisition invalide |
| Resultats instables | Trop peu d'iterations ou objectif bruite |

