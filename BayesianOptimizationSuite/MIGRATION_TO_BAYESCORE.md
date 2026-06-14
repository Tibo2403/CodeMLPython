# Plan de migration vers BayesCore

Ce document decrit comment harmoniser les projets existants autour du noyau
commun `BayesCore`.

## Etat actuel

| Projet | Etat | Action recommandee |
| --- | --- | --- |
| `BayesCore` | Noyau cree | Stabiliser API et tests |
| `SecurityBayesOptimizer` | Deja branche sur `BayesCore` | Ajouter SARIF et baseline random |
| `TokenBayesOptimizer` | Optimiseur specialise autonome | Migrer le moteur interne vers `BayesCore` |
| `HRBayesOptimizer` | Moteur historique avec contraintes | Extraire contraintes/bruit vers `BayesCore v2` |

## Objectif de migration

Tous les projets doivent partager:

- `Parameter`;
- `SearchSpace`;
- `OptimizationConfig`;
- `GenericBayesianOptimizer`;
- les fonctions d'acquisition;
- la logique Gaussian Process.
- les exports JSON/CSV generiques;
- le front de Pareto generique;
- les contraintes simples basees sur les metadonnees;
- la baseline `random_search`.
- la gestion simple du bruit avec `observation_repeats` et `gp_noise`.

Chaque projet conserve:

- sa fonction objectif;
- ses observations metier;
- ses rapports;
- sa CLI;
- ses contraintes specifiques quand elles ne sont pas encore generiques.

## Migration TokenBayesOptimizer

Etapes recommandees:

1. Importer `Parameter`, `SearchSpace`, `OptimizationConfig` et
   `GenericBayesianOptimizer` depuis `bayes_core`.
2. Garder `TokenQualityObjective` tel quel.
3. Creer un adaptateur `TokenBayesOptimizer.run()` qui appelle le noyau
   generique.
4. Convertir les `GenericObservation` en observations tokens.
5. Garder les rapports Markdown/HTML existants.
6. Lancer les tests existants sans modifier les sorties publiques.

Critere de reussite:

```text
Les tests Token passent et les rapports generes gardent les memes champs.
```

## Migration HRBayesOptimizer

`HRBayesOptimizer` contient des idees plus avancees:

- contraintes explicites;
- probabilite de faisabilite;
- gestion du bruit;
- reprise d'une optimisation existante.

Ces elements peuvent devenir une version future de `BayesCore`.

Migration prudente:

1. Ne pas remplacer directement le moteur RH.
2. Identifier les fonctions generiques dans `hr_bayes_optimizer/core.py`.
3. Ajouter des tests equivalents dans `BayesCore`.
4. Introduire `ConstrainedBayesianOptimizer` dans `BayesCore` si necessaire.
5. Brancher HR sur ce nouveau composant seulement quand les tests passent.

## Extension DrugDiscoveryBayesOptimizer

Le projet drug discovery devrait demarrer directement sur `BayesCore`.

Structure recommandee:

```text
DrugDiscoveryBayesOptimizer/
  README.md
  pyproject.toml
  examples/
  scripts/
    run_tests.py
  src/
    drug_discovery_bayes_optimizer/
      __init__.py
      cli.py
      objective.py
      reports.py
  tests/
    test_optimizer.py
```

Metriques probables:

- affinite de liaison predite;
- risque ADMET;
- toxicite;
- nouveaute;
- synthese estimable;
- diversite chimique.

## Regles de compatibilite

- Ne pas casser les CLI existantes.
- Ne pas changer les noms de fichiers exportes sans raison.
- Ajouter une note de migration dans chaque README.
- Garder des tests rapides sans dependances externes.
- Eviter d'ajouter des dependances lourdes dans `BayesCore`.

## Roadmap courte

1. Migrer `TokenBayesOptimizer`.
2. Utiliser la baseline random commune dans les projets metier.
3. Utiliser le front de Pareto commun quand les metriques sont simples.
4. Utiliser `observation_repeats` dans les projets avec mesures instables.
5. Ajouter la reprise de run depuis fichier dans les CLI metier.
6. Creer `DrugDiscoveryBayesOptimizer`.

