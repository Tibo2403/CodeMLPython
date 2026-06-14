# FAQ

## Est-ce que BayesCore remplace les projets metier ?

Non. `BayesCore` remplace seulement le moteur generique d'optimisation. Les
projets metier gardent leurs parametres, leurs fonctions objectif, leurs
metriques et leurs rapports.

## Pourquoi utiliser l'optimisation bayesienne ?

Parce qu'une recherche exhaustive teste trop de combinaisons et qu'une recherche
aleatoire peut gaspiller beaucoup d'essais. L'optimisation bayesienne apprend un
modele approximatif du score, puis choisit les essais suivants de maniere plus
informee.

## Est-ce que le meilleur score est toujours le meilleur choix ?

Pas forcement. Le meilleur score depend des poids choisis dans la fonction
objectif. Il faut aussi lire le front de Pareto, les metadonnees et valider le
resultat avec le domaine concerne.

## Quelle fonction d'acquisition choisir ?

| Strategie | Choix recommande |
| --- | --- |
| `expected_improvement` | Defaut general |
| `probability_improvement` | Quand on veut surtout battre le meilleur point connu |
| `lower_confidence_bound` | Quand on veut explorer davantage |
| `random_search` | Pour comparer avec une baseline simple |

## Combien d'iterations utiliser ?

Pour un test rapide, 10 a 30 iterations suffisent. Pour un resultat plus stable,
utilise plutot 40 a 100 iterations selon le cout de chaque evaluation.

## Pourquoi garder les metadonnees ?

Le score seul ne suffit pas. Les metadonnees expliquent pourquoi une
configuration est bonne ou mauvaise: tokens, qualite, findings, latence, cout,
toxicite, contraintes, etc.

## Pourquoi comparer avec random search ?

Cela verifie que le modele bayesien apporte vraiment quelque chose. Si la
recherche aleatoire fait aussi bien, l'espace de recherche ou la fonction
objectif doit probablement etre revue.

## Le projet drug discovery existe-t-il deja ?

Pas encore comme dossier complet. Il est documente comme prochaine application
possible de `BayesCore`. Le guide [BayesCore/NEW_PROJECT_GUIDE.md](BayesCore/NEW_PROJECT_GUIDE.md)
contient deja un exemple de fonction objectif drug discovery.

## Le scanner securite peut-il scanner n'importe quoi ?

Non. Il est concu pour un usage defensif local uniquement. Il ne doit etre lance
que sur du code que tu possedes ou que tu es autorise a auditer. Il ne fait pas
d'exploitation, pas de scan reseau et pas de brute force.

## Que faut-il ameliorer ensuite ?

Les prochaines etapes les plus utiles sont:

1. migrer `TokenBayesOptimizer` vers `BayesCore`;
2. brancher `HyperparameterBayesOptimizer` sur de vrais entrainements ML;
3. creer le projet `DrugDiscoveryBayesOptimizer`;
4. extraire les contraintes de `HRBayesOptimizer` vers une version future de
   `BayesCore`.

