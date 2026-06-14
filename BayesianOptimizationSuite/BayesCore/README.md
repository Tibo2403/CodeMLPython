# BayesCore

Noyau reutilisable d'optimisation bayesienne pour plusieurs projets metier:

- reduction du nombre de tokens d'une IA;
- priorisation de configurations de drug discovery;
- recherche defensive de failles de securite dans du code local;
- tout probleme ou chaque evaluation coute du temps, des ressources ou de
  l'argent.

Le noyau ne connait pas le domaine. Il sait seulement:

- decrire un espace de recherche avec `Parameter` et `SearchSpace`;
- tester des configurations;
- apprendre un modele surrogate avec un Gaussian Process;
- choisir le prochain essai avec une fonction d'acquisition;
- retourner un historique exploitable.

## Composants

| Composant | Role |
| --- | --- |
| `Parameter` | Decrit un parametre `float`, `int` ou `categorical` |
| `SearchSpace` | Regroupe les parametres optimisables |
| `OptimizationConfig` | Configure iterations, exploration et acquisition |
| `GenericBayesianOptimizer` | Execute la boucle d'optimisation |
| `GenericObservation` | Stocke une configuration testee et ses metadonnees |
| `Constraint` | Selectionne la meilleure observation faisable selon les metadonnees |
| `observation_repeats` | Repete une evaluation bruitee et moyenne automatiquement |
| `gp_noise` | Evite au Gaussian Process de sur-apprendre le bruit |
| `expected_improvement` | Favorise les candidats capables d'ameliorer le meilleur score |
| `probability_improvement` | Favorise les candidats ayant une forte probabilite de progres |
| `lower_confidence_bound` | Equilibre exploitation et exploration via l'incertitude |
| `random_search` | Baseline simple pour comparer le gain bayesien |

## Documentation

- [Architecture](ARCHITECTURE.md): separation entre noyau generique et projets
  metier.
- [Usage professionnel](PROFESSIONAL_USAGE.md): bonnes pratiques pour passer
  d'un prototype a un outil exploitable.
- [Guide nouveau projet](NEW_PROJECT_GUIDE.md): methode pour creer un nouvel
  optimiseur metier.
- [Reference API](API_REFERENCE.md): resume des classes et fonctions publiques.

## Contrat d'integration

Un projet metier doit fournir une fonction objectif:

```python
def objective(config):
    return objective_value, {"domain_metric": 123}
```

Le noyau minimise `objective_value`. Les metadonnees restent libres: tokens,
qualite, molecules candidates, findings de securite, latence, cout, etc.

Exemple minimal:

```python
from bayes_core import GenericBayesianOptimizer, OptimizationConfig, Parameter, SearchSpace

space = SearchSpace((Parameter("x", "float", low=0, high=1),))
optimizer = GenericBayesianOptimizer(space, objective, OptimizationConfig(iterations=20))
result = optimizer.run()
```

Fonctions communes disponibles sur le resultat:

- `to_json()` et `write_json(path)`;
- `write_csv(path)`;
- `pareto_front({"metric": "min" | "max"})`;
- `from_json(text)` pour recharger un resultat.

## Fonctionnement

1. Le noyau echantillonne quelques configurations aleatoires.
2. Chaque configuration est evaluee par la fonction objectif du projet.
3. Un Gaussian Process apprend une approximation du score.
4. La fonction d'acquisition choisit les candidats prometteurs.
5. Le cycle recommence jusqu'au budget d'iterations.
6. Le projet metier transforme les observations en rapport, dashboard ou export.

Le noyau peut aussi lancer une baseline aleatoire avec
`optimizer.run_random_baseline()`.

## Gestion simple du bruit

Si une evaluation est instable, par exemple une mesure IA, un score scientifique
ou une simulation, configure simplement:

```python
OptimizationConfig(
    iterations=30,
    observation_repeats=3,
    gp_noise=0.01,
)
```

Le noyau appelle alors la fonction objectif 3 fois pour la meme configuration,
puis ajoute automatiquement aux metadonnees:

- `repeats`;
- `objective_std`;
- `replicate_objectives`;
- `<metric>_std` pour les metriques numeriques repetees.

La fonction objectif ne change pas. Elle retourne toujours:

```python
return objective_value, metadata
```

## Cas de figure

| Projet | Ce qui change | Objectif minimise | Metadonnees typiques |
| --- | --- | --- | --- |
| Token optimizer | Prompt, contexte, RAG, format | Tokens + penalite qualite | `tokens`, `quality`, `latency_ms` |
| Drug discovery | Descripteurs molecule, filtres ADMET, docking | Risque - score activite - developpabilite | `affinity`, `toxicity`, `novelty`, `admet` |
| Security optimizer | Parametres de scan statique local | Temps/bruit - bonus findings | `findings`, `high_findings`, `precision_proxy` |

## Pourquoi separer le noyau ?

La separation evite de recopier la meme logique dans chaque projet. Les projets
peuvent evoluer sur leurs metriques, leurs rapports et leurs interfaces sans
toucher au moteur bayesien. En pratique, cela rend les tests plus propres, les
ameliorations plus rapides et les nouveaux cas d'usage plus faciles a creer.

## Tests

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore
python scripts\run_tests.py
```

