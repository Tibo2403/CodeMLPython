# Bayesian Optimization Hub

Cette page met en evidence l'ecosysteme d'optimisation bayesienne associe a
`CodeMLPython`.

L'objectif est simple: utiliser un noyau bayesien reutilisable pour optimiser
des configurations couteuses a tester en data science, IA, drug discovery,
securite defensive et hyperparametres ML.

## Vue d'ensemble

| Projet | Role | Statut |
| --- | --- | --- |
| [`BayesCore`](BayesianOptimizationSuite/BayesCore/README.md) | Noyau generique d'optimisation bayesienne | Actif |
| [`HyperparameterBayesOptimizer`](BayesianOptimizationSuite/HyperparameterBayesOptimizer/README.md) | Extension pour tuning d'hyperparametres ML | Actif |
| [`SecurityBayesOptimizer`](BayesianOptimizationSuite/SecurityBayesOptimizer/README.md) | Extension defensive pour optimiser des scans statiques locaux | Actif |
| [`TokenBayesOptimizer`](BayesianOptimizationSuite/TokenBayesOptimizer/README.md) | Optimisation du cout token/qualite des systemes IA | Prototype avance |
| [`HRBayesOptimizer`](BayesianOptimizationSuite/HRBayesOptimizer/README.md) | Optimisation RH sous contraintes | Prototype avance |
| Drug discovery bayesien | Prochaine extension naturelle pour `CodeMLPython` | A creer |

## Pourquoi c'est important pour CodeMLPython

`CodeMLPython` contient deja des notebooks ML medicaux, un pipeline de drug
discovery et des outils de statistiques. L'optimisation bayesienne ajoute une
couche experimentale au-dessus:

```text
configuration candidate
  -> evaluation ML / IA / drug discovery
  -> score metier
  -> prochaine configuration proposee par BayesCore
```

Cela permet de tester moins de configurations au hasard et de mieux documenter
les choix experimentaux.

## Cas d'usage principaux

### Hyperparametres ML

Optimiser:

- `learning_rate`;
- `max_depth`;
- `n_estimators`;
- `regularization`;
- `model_family`;
- seuil de decision.

Score typique:

```text
objective = (1 - f1) + penalite_latence + penalite_cout
```

Extension disponible:

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HyperparameterBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore\src;C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HyperparameterBayesOptimizer\src"
python -m hyperparameter_bayes_optimizer.cli --iterations 30 --observation-repeats 3 --gp-noise 0.01
```

### Drug discovery

Optimiser a terme:

- poids entre activite predite et drug-likeness;
- seuils de fiabilite conformal prediction;
- contraintes ADMET;
- filtres Lipinski/Veber;
- diversite des candidats;
- cout de generation et de validation.

Score possible:

```text
objective = toxicite + risque_admet - bonus_activite - bonus_fiabilite - bonus_drug_likeness
```

### IA et reduction de tokens

Optimiser:

- taille du contexte;
- `top_k` retrieval;
- compression;
- nombre d'exemples few-shot;
- style de sortie;
- qualite minimale.

Extension disponible:

[`TokenBayesOptimizer`](BayesianOptimizationSuite/TokenBayesOptimizer/README.md)

### Securite defensive

Optimiser les reglages d'un scan local autorise:

- profondeur de scan;
- profil de regles;
- seuils;
- temps d'analyse;
- bruit des findings.

Extension disponible:

[`SecurityBayesOptimizer`](BayesianOptimizationSuite/SecurityBayesOptimizer/README.md)

## Capacites du noyau BayesCore

`BayesCore` fournit maintenant:

- espaces de recherche typÃ©s: `float`, `int`, `categorical`;
- Gaussian Process;
- fonctions d'acquisition;
- baseline `random_search`;
- contraintes simples;
- front de Pareto;
- exports JSON/CSV;
- reprise depuis observations existantes;
- gestion simple du bruit avec `observation_repeats` et `gp_noise`.

## Documentation utile

- [Quickstart bayesien](BAYESIAN_QUICKSTART.md)
- [Vue globale des projets](BayesianOptimizationSuite/BAYESIAN_OPTIMIZATION_PROJECTS.md)
- [Architecture BayesCore](BayesianOptimizationSuite/BayesCore/ARCHITECTURE.md)
- [Reference API BayesCore](BayesianOptimizationSuite/BayesCore/API_REFERENCE.md)
- [Guide nouveau projet](BayesianOptimizationSuite/BayesCore/NEW_PROJECT_GUIDE.md)
- [Plan de migration](BayesianOptimizationSuite/MIGRATION_TO_BAYESCORE.md)
- [Roadmap](BayesianOptimizationSuite/ROADMAP.md)

## Prochaine integration recommandee dans CodeMLPython

1. Ajouter une extension drug discovery bayesienne basee sur `BayesCore`.
2. Brancher `HyperparameterBayesOptimizer` sur les notebooks medicaux.
3. Ajouter un exemple d'optimisation de seuil de classification medicale.
4. Ajouter une comparaison `random_search` vs bayesien.
5. Exporter les resultats experimentaux en JSON/CSV pour reproductibilite.

