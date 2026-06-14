# Projets d'optimisation bayesienne

Ce dossier regroupe plusieurs projets bases sur une meme idee: utiliser
l'optimisation bayesienne pour trouver de meilleurs reglages avec peu
d'evaluations.

## Par ou commencer ?

| Besoin | Document |
| --- | --- |
| Comprendre la vision globale | [Vue d'ensemble](BAYESIAN_OPTIMIZATION_PROJECTS.md) |
| Lancer rapidement un exemple | [Quickstart](QUICKSTART.md) |
| Comprendre le noyau commun | [BayesCore](BayesCore/README.md) |
| Creer un nouveau projet metier | [Guide nouveau projet](BayesCore/NEW_PROJECT_GUIDE.md) |
| Connaitre les classes disponibles | [Reference API](BayesCore/API_REFERENCE.md) |
| Planifier les prochaines etapes | [Roadmap](ROADMAP.md) |
| Migrer les projets vers le noyau | [Plan de migration](MIGRATION_TO_BAYESCORE.md) |
| Repondre aux questions courantes | [FAQ](FAQ.md) |

## Projets

| Projet | Description |
| --- | --- |
| `BayesCore` | Noyau generique d'optimisation bayesienne |
| `SecurityBayesOptimizer` | Optimisation defensive de scans statiques locaux |
| `TokenBayesOptimizer` | Reduction des tokens IA sous contrainte de qualite |
| `HyperparameterBayesOptimizer` | Optimisation d'hyperparametres ML |
| `HRBayesOptimizer` | Optimisation RH sous contraintes de cout, qualite, equite et bien-etre |

## Architecture cible

```text
BayesCore
  -> SecurityBayesOptimizer
  -> TokenBayesOptimizer
  -> HyperparameterBayesOptimizer
  -> HRBayesOptimizer
  -> futurs projets metier
```

Le noyau commun gere la boucle bayesienne. Les projets metier definissent leurs
parametres, leur fonction objectif, leurs rapports et leurs contraintes.

## Commandes rapides

Tester le noyau:

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore
python scripts\run_tests.py
```

Lancer la demo securite:

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\SecurityBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore\src;C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\SecurityBayesOptimizer\src"
python -m security_bayes_optimizer.cli --target examples\demo_project --iterations 20
```

Lancer la demo tokens:

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\TokenBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\TokenBayesOptimizer\src"
python -m token_bayes_optimizer.cli --iterations 24 --quality-floor 0.82
```

Lancer la demo hyperparametres:

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HyperparameterBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore\src;C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HyperparameterBayesOptimizer\src"
python -m hyperparameter_bayes_optimizer.cli --iterations 30 --observation-repeats 3 --gp-noise 0.01
```

## Etat actuel

- `BayesCore` existe et possede ses tests.
- `SecurityBayesOptimizer` est branche sur `BayesCore`.
- `HyperparameterBayesOptimizer` est branche sur `BayesCore`.
- `TokenBayesOptimizer` reste specialise, avec une migration documentee.
- `HRBayesOptimizer` contient des idees avancees a extraire plus tard:
  contraintes, bruit et reprise de run.

## Principe commun

Chaque projet suit le meme schema:

```text
configuration candidate
  -> evaluation metier
  -> score a minimiser
  -> metadonnees explicatives
  -> prochaine configuration choisie par BayesCore
```

La qualite du projet depend surtout de la fonction objectif. Le noyau aide a
chercher efficacement, mais il ne remplace pas la validation metier.

