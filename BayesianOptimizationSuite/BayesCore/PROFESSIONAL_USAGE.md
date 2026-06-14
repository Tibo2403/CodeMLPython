# Usage professionnel du noyau bayesien

Ce guide sert a transformer les prototypes en outils exploitables par une
equipe.

## Definition du probleme

Avant de lancer l'optimisation, documente:

- la decision que l'outil doit aider a prendre;
- les parametres vraiment controlables;
- le cout d'une evaluation;
- les contraintes non negociables;
- les metriques de succes.

Exemples:

| Domaine | Decision optimisee | Contrainte critique |
| --- | --- | --- |
| Tokens IA | Choisir une configuration prompt/RAG | Qualite minimale |
| Drug discovery | Prioriser des candidats molecule | Toxicite et developpabilite |
| Securite | Choisir une configuration de scan | Usage defensif autorise |

## Construire une bonne fonction objectif

Une fonction objectif professionnelle doit etre lisible et justifiable:

```text
objective = cout + penalites - bonus
```

Bonnes pratiques:

- utiliser des unites coherentes;
- separer les bonus et penalites dans les metadonnees;
- eviter les poids magiques non documentes;
- tester plusieurs poids avec une baseline;
- conserver la configuration et les resultats bruts.

## Metriques minimales a exporter

Chaque run devrait produire:

| Champ | Pourquoi |
| --- | --- |
| `best.config` | Rejouer la meilleure configuration |
| `best.objective` | Comprendre le score optimise |
| `observations` | Auditer toutes les evaluations |
| `optimization_config` | Reproduire le run |
| `metadata` | Expliquer les compromis metier |

## Validation

Une optimisation bayesienne propose des candidats, elle ne remplace pas la
validation metier. Avant d'utiliser une configuration en production:

1. relancer la meilleure configuration;
2. comparer avec une baseline simple;
3. verifier le front de Pareto;
4. inspecter les cas d'echec;
5. faire valider les resultats par le domaine concerne.

## Gerer le bruit simplement

Si une evaluation varie d'un appel a l'autre, active les repetitions:

```python
OptimizationConfig(observation_repeats=3, gp_noise=0.01)
```

Utilise `observation_repeats` quand:

- la qualite IA depend d'un modele non deterministe;
- une simulation contient du hasard;
- une mesure de latence varie;
- un score metier vient d'un echantillon de donnees.

Commence avec `3` repetitions. Monte a `5` si les resultats restent instables.
Evite de repeter trop fortement si chaque evaluation coute cher.

## Risques courants

| Risque | Symptome | Correction |
| --- | --- | --- |
| Score mal formule | Le meilleur resultat est inutilisable | Ajouter une penalite ou une contrainte |
| Trop peu d'iterations | Resultats instables | Augmenter `iterations` et `initial_points` |
| Espace trop large | L'optimiseur explore sans converger | Resserrer les bornes |
| Metrique simulee | Gain theorique non confirme | Brancher une evaluation reelle |
| Sur-optimisation | Bon score, mauvais usage reel | Evaluer sur plusieurs cas representatifs |
| Bruit ignore | Le meilleur point change souvent | Activer `observation_repeats` et `gp_noise` |

## Checklist avant livraison

- README avec installation, exemples et limites.
- Tests rapides sans dependances lourdes.
- Run reproductible avec `random_state`.
- Exports JSON et CSV.
- Rapport lisible pour les non-developpeurs.
- Baseline `random_search` ou configuration conservative.
- Limites et hypotheses explicites.
- Donnees sensibles exclues des rapports publics.

