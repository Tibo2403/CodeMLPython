# Hyperparameter Bayes Optimizer

Extension de `BayesCore` pour optimiser des hyperparametres de modeles ML.

Elle sert a choisir automatiquement de bons reglages quand chaque evaluation
coute du temps: entrainement, validation, inference, cout GPU ou appels API.

## Ce que l'extension optimise

Preset inclus pour classification tabulaire:

| Hyperparametre | Type | Role |
| --- | --- | --- |
| `learning_rate` | float | Vitesse d'apprentissage |
| `max_depth` | int | Complexite des arbres |
| `n_estimators` | int | Nombre d'arbres/estimateurs |
| `regularization` | float | Penalisation du modele |
| `subsample` | float | Sous-echantillonnage |
| `model_family` | categoriel | Famille de modele |
| `decision_threshold` | float | Seuil de classification |

## Fonction objectif

Le noyau minimise un score. Pour les hyperparametres, un score typique est:

```text
objective = (1 - f1) + penalite_latence + penalite_cout + penalite_qualite
```

La demo inclut `SimulatedModelObjective`. Dans un vrai projet, remplace-la par
une fonction qui entraine et valide ton modele.

## Lancer la demo

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HyperparameterBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore\src;C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HyperparameterBayesOptimizer\src"
python -m hyperparameter_bayes_optimizer.cli --iterations 30
```

Avec gestion simple du bruit:

```powershell
python -m hyperparameter_bayes_optimizer.cli `
  --iterations 40 `
  --observation-repeats 3 `
  --gp-noise 0.01
```

Avec contrainte de latence:

```powershell
python -m hyperparameter_bayes_optimizer.cli `
  --iterations 40 `
  --min-f1 0.82 `
  --max-latency-ms 70
```

## Sorties

- `hyperparameter_optimization_result.json`: resultat complet.
- `hyperparameter_optimization_history.csv`: historique des essais.
- `hyperparameter_optimization_report.html`: rapport visuel.

## Brancher un vrai modele

Remplace `SimulatedModelObjective` par une fonction de ce type:

```python
def train_and_validate_objective(config):
    model = build_model(
        learning_rate=float(config["learning_rate"]),
        max_depth=int(config["max_depth"]),
        n_estimators=int(config["n_estimators"]),
    )
    metrics = train_and_validate(model)
    objective = 1.0 - metrics["f1"] + 0.0008 * metrics["latency_ms"]
    return objective, metrics
```

Puis:

```python
from bayes_core import OptimizationConfig
from hyperparameter_bayes_optimizer import HyperparameterBayesOptimizer, HyperparameterSearchSpace


optimizer = HyperparameterBayesOptimizer(
    search_space=HyperparameterSearchSpace.tabular_classification_space(),
    objective=train_and_validate_objective,
    config=OptimizationConfig(iterations=40, observation_repeats=3, gp_noise=0.01),
)
result = optimizer.run()
```

## Tests

```powershell
python scripts\run_tests.py
```

