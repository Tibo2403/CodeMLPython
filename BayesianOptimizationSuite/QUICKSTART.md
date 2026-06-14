# Quickstart

Ce guide permet de tester les projets rapidement depuis PowerShell.

## 1. Tester BayesCore

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore
python scripts\run_tests.py
```

Resultat attendu:

```text
2 passed, 0 failed
```

Si `python` n'est pas disponible:

```powershell
C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\run_tests.py
```

## 2. Lancer SecurityBayesOptimizer

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\SecurityBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore\src;C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\SecurityBayesOptimizer\src"
python -m security_bayes_optimizer.cli --target examples\demo_project --iterations 20
```

Fichiers generes:

- `security_optimization_result.json`
- `security_optimization_history.csv`
- `security_optimization_report.html`

L'outil securite est defensif uniquement. Il doit etre lance seulement sur du
code local que tu possedes ou que tu es autorise a auditer.

## 3. Lancer TokenBayesOptimizer

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\TokenBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\TokenBayesOptimizer\src"
python -m token_bayes_optimizer.cli --iterations 24 --quality-floor 0.82
```

Avec comparaison random search:

```powershell
python -m token_bayes_optimizer.cli --iterations 40 --compare-random
```

Fichiers generes:

- `token_optimization_result.json`
- `token_optimization_history.csv`
- `token_optimization_report.md`
- `token_optimization_report.html`

## 4. Lancer HyperparameterBayesOptimizer

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HyperparameterBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore\src;C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HyperparameterBayesOptimizer\src"
python -m hyperparameter_bayes_optimizer.cli --iterations 30 --observation-repeats 3 --gp-noise 0.01
```

Fichiers generes:

- `hyperparameter_optimization_result.json`
- `hyperparameter_optimization_history.csv`
- `hyperparameter_optimization_report.html`

## 5. Lancer HRBayesOptimizer

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HRBayesOptimizer
$env:PYTHONPATH="src"
python -m hr_bayes_optimizer.cli --iterations 35
```

Fichiers generes:

- `hr_optimization_result.json`
- `hr_optimization_history.csv`
- `hr_optimization_report.md`
- `hr_optimization_report.html`

## 6. Lire les resultats

Commence par le rapport HTML de chaque projet. Ensuite, ouvre le JSON si tu veux
inspecter la configuration exacte, les metadonnees et l'historique complet.

## 7. Modifier un projet

Pour creer un nouveau domaine:

1. lis [BayesCore/NEW_PROJECT_GUIDE.md](BayesCore/NEW_PROJECT_GUIDE.md);
2. definis un `SearchSpace`;
3. ecris une fonction objectif;
4. lance `GenericBayesianOptimizer`;
5. ajoute des exports JSON/CSV/HTML;
6. ajoute un test rapide.

