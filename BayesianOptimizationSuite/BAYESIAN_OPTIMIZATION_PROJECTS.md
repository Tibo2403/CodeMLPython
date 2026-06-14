# Portefeuille de projets d'optimisation bayesienne

Ce dossier contient plusieurs projets qui appliquent le meme principe:
utiliser l'optimisation bayesienne pour choisir de bons reglages quand chaque
evaluation coute du temps, de l'argent ou de l'expertise.

## Vue d'ensemble

| Projet | Domaine | Statut du noyau | Role |
| --- | --- | --- | --- |
| `BayesCore` | Generique | Noyau partage | Moteur bayesien reutilisable |
| `SecurityBayesOptimizer` | Securite defensive | Branche sur `BayesCore` | Optimiser des scans statiques locaux |
| `HyperparameterBayesOptimizer` | Data science / ML | Branche sur `BayesCore` | Optimiser des hyperparametres |
| `TokenBayesOptimizer` | IA / tokens | Metier specialise | Reduire les tokens en gardant la qualite |
| `HRBayesOptimizer` | RH | Moteur avance historique | Optimiser des politiques RH sous contraintes |

## Architecture cible

L'architecture cible est:

```text
BayesCore
  -> TokenBayesOptimizer
  -> SecurityBayesOptimizer
  -> HyperparameterBayesOptimizer
  -> DrugDiscoveryBayesOptimizer
  -> HRBayesOptimizer
```

`BayesCore` doit rester petit, generique et testable. Les projets metier
gardent leurs simulateurs, rapports, contraintes, CLI et exports.

## Ce que le noyau resout

Le noyau commun gere:

- l'echantillonnage initial;
- le modele Gaussian Process;
- les fonctions d'acquisition;
- la selection du prochain candidat;
- l'historique des observations;
- la reproductibilite par `random_state`.

Il ne gere pas:

- la signification metier des parametres;
- la validation humaine;
- les rapports specialises;
- les contraintes legales, securite ou ethique;
- les integrations externes.

## Comparaison des fonctions objectif

| Domaine | Exemple de score minimise |
| --- | --- |
| Tokens IA | `tokens + penalite_si_qualite_trop_basse` |
| Securite | `cout_temps - bonus_findings - bonus_high - bonus_precision` |
| Hyperparametres ML | `(1 - f1) + penalite_latence + penalite_cout` |
| Drug discovery | `toxicite + risque_admet - bonus_affinite - bonus_nouveaute` |
| RH | `cout + penalite_delai + penalites_contraintes` |

Le point commun est que chaque domaine transforme un compromis complexe en un
score numerique auditable.

## Priorites recommandees

1. Garder `BayesCore` stable et bien teste.
2. Migrer progressivement `TokenBayesOptimizer` vers `BayesCore`.
3. Ajouter un vrai projet `DrugDiscoveryBayesOptimizer` base sur le noyau.
4. Extraire les idees avancees de `HRBayesOptimizer`, notamment contraintes et
   bruit, vers une future version de `BayesCore`.
5. Ajouter un tableau comparatif des resultats entre random search et bayesien.

## Documentation principale

- [README racine](README.md)
- [Quickstart](QUICKSTART.md)
- [FAQ](FAQ.md)
- [BayesCore README](BayesCore/README.md)
- [Architecture BayesCore](BayesCore/ARCHITECTURE.md)
- [Usage professionnel](BayesCore/PROFESSIONAL_USAGE.md)
- [Guide nouveau projet](BayesCore/NEW_PROJECT_GUIDE.md)
- [Reference API BayesCore](BayesCore/API_REFERENCE.md)
- [Plan de migration](MIGRATION_TO_BAYESCORE.md)
- [Roadmap](ROADMAP.md)
- [Token Bayes Optimizer](TokenBayesOptimizer/README.md)
- [Security Bayes Optimizer](SecurityBayesOptimizer/README.md)
- [Hyperparameter Bayes Optimizer](HyperparameterBayesOptimizer/README.md)
- [HR Bayes Optimizer](HRBayesOptimizer/README.md)

