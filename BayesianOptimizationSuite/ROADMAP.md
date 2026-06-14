# Roadmap optimisation bayesienne

Cette roadmap organise les prochaines ameliorations par impact.

## Court terme

| Priorite | Amelioration | Impact |
| --- | --- | --- |
| P1 | Ajouter un README racine et un quickstart | Point d'entree clair |
| P1 | Migrer `TokenBayesOptimizer` vers `BayesCore` | Supprime la duplication |
| P1 | Ajouter une baseline `random_search` commune | Fait |
| P1 | Ajouter exports JSON/CSV et Pareto generiques | Fait |
| P1 | Ajouter contraintes simples dans `BayesCore` | Fait |
| P1 | Simplifier la gestion du bruit dans `BayesCore` | Fait |
| P1 | Ajouter tests de non-regression CLI | Evite les ruptures |
| P2 | Ajouter exports standardises | Facilite analyse et comparaison |
| P2 | Ajouter exemples drug discovery | Prepare le nouveau domaine |
| P2 | Ajouter extension hyperparametres ML | Fait |

## Moyen terme

| Priorite | Amelioration | Impact |
| --- | --- | --- |
| P1 | Contraintes generiques dans `BayesCore` | Reutilise les idees RH |
| P2 | Gestion du bruit d'observation | Resultats plus robustes |
| P2 | Reprise de run generique | Runs longs et incrementaux |
| P2 | Rapport HTML commun minimal | Meilleure lisibilite |

## Long terme

| Priorite | Amelioration | Impact |
| --- | --- | --- |
| P1 | Dashboard interactif | Usage non technique |
| P2 | SQLite pour experiences | Historique exploitable |
| P2 | Comparaison multi-runs | Evaluation experimentale |
| P3 | Integration CI | Usage professionnel continu |

## Definition of Done

Une amelioration est terminee quand:

- elle est documentee;
- elle a au moins un test;
- elle garde les CLI existantes fonctionnelles;
- elle produit un exemple reproductible;
- elle n'ajoute pas de dependance lourde inutile au noyau.

