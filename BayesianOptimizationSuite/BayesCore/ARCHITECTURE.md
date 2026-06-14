# Architecture BayesCore

Ce document explique comment utiliser `BayesCore` comme noyau commun pour des
projets differents sans melanger la logique bayesienne et la logique metier.

## Idee generale

`BayesCore` optimise une fonction couteuse a evaluer. Le noyau ne sait pas si
la configuration represente un prompt IA, une experience de drug discovery ou
un scan de securite. Il manipule seulement trois concepts:

| Concept | Responsabilite |
| --- | --- |
| Espace de recherche | Decrit les parametres possibles |
| Fonction objectif | Transforme une configuration en score numerique |
| Historique | Stocke les observations et leurs metadonnees |

Le score est toujours minimise. Si un projet veut maximiser une mesure, il doit
la convertir en bonus negatif dans sa fonction objectif.

## Flux d'execution

```text
SearchSpace
  -> configurations candidates
  -> objective(config)
  -> objective_value + metadata
  -> Gaussian Process
  -> acquisition function
  -> prochaine configuration
  -> resultats et rapports metier
```

## Separation des responsabilites

`BayesCore` doit contenir:

- les types generiques `Parameter`, `SearchSpace`, `OptimizationConfig`;
- le `GenericBayesianOptimizer`;
- le Gaussian Process;
- les fonctions d'acquisition;
- la serialisation minimale des observations.
- les contraintes simples basees sur les metadonnees;
- la gestion simple du bruit par repetitions d'evaluation;
- les exports generiques JSON/CSV;
- le front de Pareto generique.

Les projets metier doivent contenir:

- leurs presets d'espace de recherche;
- leurs fonctions objectif;
- leurs metriques specifiques;
- leurs rapports et exports;
- leurs interfaces CLI ou UI.

Cette separation permet de tester le moteur une seule fois et de reutiliser le
meme comportement dans plusieurs applications.

## Fonctions d'acquisition

| Strategie | Quand l'utiliser |
| --- | --- |
| `expected_improvement` | Bon choix par defaut quand le budget d'essais est limite |
| `probability_improvement` | Utile quand on veut surtout battre le meilleur score actuel |
| `lower_confidence_bound` | Utile quand il faut explorer davantage des zones incertaines |
| `random_search` | Baseline pour verifier que le bayesien apporte un gain |

## Contraintes

Les contraintes generiques sont volontairement simples. Elles lisent une valeur
dans `metadata` et verifient un seuil:

```python
Constraint("quality", "min", 0.85)
Constraint("toxicity", "max", 0.20)
```

Le noyau utilise ces contraintes pour choisir `result.best`. La fonction
d'acquisition reste volontairement simple dans cette version; une future version
pourra integrer une probabilite de faisabilite comme dans `HRBayesOptimizer`.

## Bruit d'observation

La gestion du bruit est volontairement simple:

```python
OptimizationConfig(observation_repeats=3, gp_noise=0.01)
```

Pour chaque configuration, le noyau repete l'appel a la fonction objectif,
moyenne les objectifs et conserve l'ecart-type. Le Gaussian Process utilise cet
ecart-type comme bruit d'observation, ce qui reduit le risque de choisir un
point uniquement parce qu'une mesure a ete chanceuse.

Cette approche suffit pour la plupart des projets metier. Les modeles plus
avances, par exemple contraintes probabilistes ou bruit heteroscedastique
detaille, peuvent etre ajoutes plus tard sans changer le contrat de base.

## Exemple d'adaptateur metier

Un projet metier peut envelopper le resultat generique pour ajouter ses propres
champs:

```python
from dataclasses import dataclass
from bayes_core import GenericBayesianOptimizer, OptimizationConfig


@dataclass(frozen=True)
class DomainObservation:
    objective: float
    metric: float
    metadata: dict


class DomainOptimizer:
    def __init__(self, search_space, objective):
        self.search_space = search_space
        self.objective = objective

    def run(self):
        result = GenericBayesianOptimizer(
            self.search_space,
            self.objective,
            OptimizationConfig(iterations=30),
        ).run()
        return [
            DomainObservation(
                objective=item.objective,
                metric=float(item.metadata["metric"]),
                metadata=item.metadata,
            )
            for item in result.observations
        ]
```

## Bonnes pratiques

- Garde la fonction objectif deterministe quand c'est possible.
- Ajoute `random_state` pour rendre les experiences reproductibles.
- Compare toujours avec `random_search`.
- Stocke les metadonnees utiles pour expliquer le resultat.
- Ne mets pas de logique metier dans `BayesCore`.
- Garde les rapports dans les projets metier.

## Ajouter un nouveau projet

1. Definir les parametres a optimiser.
2. Ecrire une fonction objectif qui retourne `(score, metadata)`.
3. Lancer `GenericBayesianOptimizer`.
4. Convertir les observations en objet metier.
5. Ajouter des exports lisibles: JSON, CSV, Markdown ou HTML.
6. Ajouter un test qui verifie que l'optimiseur tourne sur un petit budget.

