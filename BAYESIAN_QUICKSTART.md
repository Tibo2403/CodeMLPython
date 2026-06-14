# Quickstart bayesien

Ce guide lance rapidement les projets bayesiens depuis l'environnement local.

## 1. Tester le noyau

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore
python scripts\run_tests.py
```

Avec le Python embarque Codex:

```powershell
C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\run_tests.py
```

## 2. Optimiser des hyperparametres ML

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HyperparameterBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore\src;C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HyperparameterBayesOptimizer\src"
python -m hyperparameter_bayes_optimizer.cli --iterations 30 --observation-repeats 3 --gp-noise 0.01
```

Sorties:

- `hyperparameter_optimization_result.json`
- `hyperparameter_optimization_history.csv`
- `hyperparameter_optimization_report.html`

## 3. Optimiser un scan securite defensif

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\SecurityBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore\src;C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\SecurityBayesOptimizer\src"
python -m security_bayes_optimizer.cli --target examples\demo_project --iterations 20
```

## 4. Optimiser les tokens IA

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\TokenBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\TokenBayesOptimizer\src"
python -m token_bayes_optimizer.cli --iterations 24 --quality-floor 0.82
```

## 5. Lire les resultats

Ouvre d'abord le rapport HTML. Ensuite, utilise le CSV pour comparer les essais
et le JSON pour rejouer ou auditer la meilleure configuration.

## 6. Brancher un vrai modele CodeMLPython

La fonction objectif doit retourner:

```python
return objective_value, metadata
```

Exemple pour un notebook medical:

```python
def objective(config):
    metrics = train_medical_classifier(
        learning_rate=float(config["learning_rate"]),
        max_depth=int(config["max_depth"]),
        decision_threshold=float(config["decision_threshold"]),
    )
    objective_value = 1.0 - metrics["f1"] + 0.0008 * metrics["latency_ms"]
    return objective_value, metrics
```

Si les scores varient selon le seed ou l'echantillon:

```python
OptimizationConfig(
    iterations=40,
    observation_repeats=3,
    gp_noise=0.01,
)
```

