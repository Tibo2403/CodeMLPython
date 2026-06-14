# Token Bayes Optimizer

Projet d'optimisation bayesienne pour reduire les tokens utilises par une IA
tout en conservant un niveau de qualite minimal.

Ce projet est le cas d'usage "tokens" du noyau bayesien commun. Le noyau
reutilisable vit dans `C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore`; ce projet conserve
ses metriques et rapports specialises tokens/qualite.

Documentation commune:

- [Vue d'ensemble des projets](..\BAYESIAN_OPTIMIZATION_PROJECTS.md)
- [Architecture BayesCore](..\BayesCore\ARCHITECTURE.md)
- [Usage professionnel BayesCore](..\BayesCore\PROFESSIONAL_USAGE.md)

L'objectif n'est pas seulement de trouver le prompt le plus court. L'objectif
est de trouver la configuration la moins couteuse qui reste suffisamment bonne.

## Ce que le projet optimise

L'outil cherche automatiquement les meilleurs reglages de prompt, contexte et
sortie:

| Parametre | Description | Impact attendu |
| --- | --- | --- |
| `max_context_chars` | Taille maximale du contexte injecte | Plus grand = plus de tokens, souvent plus de qualite |
| `retrieval_top_k` | Nombre de documents recuperes dans un RAG | Plus grand = plus d'information, plus cher |
| `summary_ratio` | Ratio de compression du contexte | Plus bas = moins de tokens, risque de perte d'information |
| `few_shot_examples` | Nombre d'exemples dans le prompt | Ameliore la qualite, augmente les tokens |
| `reasoning_level` | Niveau de raisonnement demande | Plus eleve = plus qualitatif, plus couteux |
| `format_style` | Style de sortie | `compact` reduit les tokens, `verbose` les augmente |

## Principe

La fonction objectif penalise fortement les configurations qui passent sous un
seuil de qualite:

```text
objectif = tokens + penalite_qualite
```

Une configuration tres courte mais mauvaise n'est donc pas selectionnee. Le
meilleur resultat est choisi parmi les configurations qui respectent le seuil
de qualite quand il en existe au moins une.

## Fonctionnement interne

1. Le projet definit un espace de recherche avec plusieurs reglages possibles.
2. Il teste quelques configurations aleatoires pour obtenir des observations.
3. Il entraine un petit modele Gaussian Process sur les resultats observes.
4. Il utilise Expected Improvement pour proposer la prochaine configuration.
5. Il repete ce cycle jusqu'au nombre d'iterations demande.
6. Il compare la meilleure configuration avec une baseline conservative.
7. Il calcule le front de Pareto tokens/qualite pour afficher les meilleurs
   compromis.

La baseline par defaut est volontairement large:

```json
{
  "max_context_chars": 8000,
  "retrieval_top_k": 8,
  "summary_ratio": 1.0,
  "few_shot_examples": 3,
  "reasoning_level": "medium",
  "format_style": "balanced"
}
```

## Structure du projet

```text
TokenBayesOptimizer/
  README.md
  pyproject.toml
  examples/
    run_demo.ps1
    tasks.jsonl
  scripts/
    run_tests.py
  src/
    token_bayes_optimizer/
      __init__.py
      cli.py
      optimizer.py
  tests/
    test_optimizer.py
```

## Installation

Depuis le dossier du projet:

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\TokenBayesOptimizer
```

Mode sans installation, en definissant `PYTHONPATH`:

```powershell
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\TokenBayesOptimizer\src"
python -m token_bayes_optimizer.cli --iterations 24 --quality-floor 0.82
```

Mode package editable:

```powershell
pip install -e .
token-bayes-optimize --iterations 24 --quality-floor 0.82
```

## Commandes utiles

Run standard:

```powershell
python -m token_bayes_optimizer.cli --iterations 24 --quality-floor 0.82
```

Run multi-taches avec le fichier d'exemple:

```powershell
python -m token_bayes_optimizer.cli --iterations 35 --quality-floor 0.82 --tasks examples\tasks.jsonl
```

Run plus long:

```powershell
python -m token_bayes_optimizer.cli --iterations 60 --initial-points 12 --quality-floor 0.85
```

Changer la strategie d'acquisition bayesienne:

```powershell
python -m token_bayes_optimizer.cli `
  --tasks examples\tasks.jsonl `
  --acquisition lower_confidence_bound `
  --exploration-weight 1.8 `
  --iterations 40
```

Comparer avec une recherche aleatoire au meme budget:

```powershell
python -m token_bayes_optimizer.cli `
  --tasks examples\tasks.jsonl `
  --iterations 40 `
  --compare-random
```

Changer les fichiers de sortie:

```powershell
python -m token_bayes_optimizer.cli `
  --iterations 40 `
  --json-output results.json `
  --csv-output history.csv `
  --report-output report.md `
  --html-output report.html
```

Desactiver la baseline:

```powershell
python -m token_bayes_optimizer.cli --no-baseline
```

## Options CLI

| Option | Defaut | Role |
| --- | ---: | --- |
| `--iterations` | `30` | Nombre total de configurations testees |
| `--initial-points` | `8` | Nombre de points aleatoires au depart |
| `--quality-floor` | `0.82` | Qualite minimale acceptable |
| `--quality-penalty` | `30000` | Penalite appliquee si la qualite est trop basse |
| `--random-state` | `42` | Seed pour rendre les runs reproductibles |
| `--acquisition` | `expected_improvement` | Strategie bayesienne: `expected_improvement`, `probability_improvement`, `lower_confidence_bound` |
| `--exploration-weight` | `1.25` | Force d'exploration pour `lower_confidence_bound` |
| `--min-candidate-distance` | `0.04` | Penalise les candidats trop proches des essais precedents |
| `--no-baseline` | off | Ne pas evaluer la baseline conservative |
| `--tasks` | vide | Fichier JSONL de taches pour evaluation multi-prompts |
| `--compare-random` | off | Lance aussi une baseline random search |
| `--json-output` | `token_optimization_result.json` | Export complet en JSON |
| `--csv-output` | `token_optimization_history.csv` | Historique des evaluations |
| `--report-output` | `token_optimization_report.md` | Rapport Markdown |
| `--html-output` | `token_optimization_report.html` | Rapport HTML visuel |

## Fichiers generes

Apres un run, le projet produit:

- `token_optimization_result.json`: meilleure configuration, baseline, savings
  et historique complet.
- `token_optimization_history.csv`: historique lisible dans Excel, Power BI ou
  pandas.
- `token_optimization_report.md`: rapport texte avec savings, meilleure config,
  top evaluations et front de Pareto.
- `token_optimization_report.html`: rapport visuel avec graphique tokens/qualite.

## Evaluation multi-taches

Le fichier [examples/tasks.jsonl](examples/tasks.jsonl) permet d'evaluer une
configuration sur plusieurs cas representatifs:

- question/reponse support;
- reponse RAG avec documents;
- resume long;
- revue de code;
- extraction JSON.

Chaque ligne JSONL decrit une tache:

```json
{"name":"code_review","kind":"code","prompt":"Review a code diff...","quality_weight":1.4,"token_multiplier":1.15,"requires_examples":true,"requires_reasoning":true}
```

Champs disponibles:

| Champ | Role |
| --- | --- |
| `name` | Identifiant lisible de la tache |
| `kind` | Type: `qa`, `rag`, `summarization`, `code`, `extraction` |
| `prompt` | Prompt representatif |
| `quality_weight` | Importance relative dans le score moyen |
| `token_multiplier` | Cout relatif de la tache |
| `requires_retrieval` | Penalise les configs avec trop peu de documents |
| `requires_examples` | Penalise les configs sans exemples few-shot |
| `requires_reasoning` | Penalise les configs avec raisonnement trop faible |

Le JSON de sortie inclut alors un detail par tache dans `metadata.tasks`, ce qui
permet de voir quelle famille de prompts tire la qualite vers le bas.

## Lire les resultats

Exemple de sortie:

```text
Best configuration
  tokens: 1650
  quality: 0.8346
  objective: 1650.0000
  baseline tokens: 3870
  tokens saved: 2220 (57.36%)
```

Interpretation:

- `tokens`: tokens estimes pour la configuration optimisee.
- `quality`: score de qualite estime entre `0` et `1`.
- `objective`: score minimise par l'optimiseur.
- `baseline tokens`: cout de la configuration conservative.
- `tokens saved`: economie estimee par rapport a la baseline.

Le rapport HTML est le plus pratique pour explorer le compromis tokens/qualite:

[token_optimization_report.html](./token_optimization_report.html)

## Workflow recommande

1. Choisir un petit jeu de taches representatif.
2. Lancer un run court pour verifier que les metriques reagissent correctement.
3. Lancer un run plus long avec `--compare-random`.
4. Inspecter le front de Pareto.
5. Choisir une configuration qui respecte la qualite minimale.
6. Rejouer cette configuration sur des cas non vus.
7. Mesurer le gain reel de tokens et de cout.

## Front de Pareto

Le front de Pareto contient les configurations non dominees:

- aucune autre configuration n'utilise moins ou autant de tokens avec une
  qualite superieure ou egale;
- aucune autre configuration n'a une meilleure qualite avec autant ou moins de
  tokens.

C'est utile quand la meilleure configuration automatique n'est pas celle que tu
veux choisir en pratique. Par exemple, tu peux accepter 200 tokens de plus pour
gagner beaucoup en qualite.

## Tests

Le projet inclut un runner sans dependance externe:

```powershell
python scripts/run_tests.py
```

Il verifie:

- la fonction objectif tokens/qualite;
- le modele Gaussian Process;
- Expected Improvement;
- la selection d'une meilleure configuration valide;
- les exports CSV, Markdown, HTML;
- le front de Pareto.

## Utilisation dans un vrai systeme IA

Aujourd'hui, `TokenQualityObjective` simule les tokens et la qualite. Pour un
usage professionnel, il faut remplacer cette fonction par une evaluation reelle:

1. construire un prompt avec la configuration candidate;
2. appeler le modele IA;
3. mesurer les tokens reels;
4. evaluer la reponse avec des tests, un score humain ou un LLM judge;
5. retourner l'objectif et les metadonnees.

Exemple:

```python
from token_bayes_optimizer import SearchSpace, TokenBayesOptimizer, TokenOptimizationConfig


def real_objective(config):
    prompt = build_prompt(
        max_context_chars=int(config["max_context_chars"]),
        retrieval_top_k=int(config["retrieval_top_k"]),
        summary_ratio=float(config["summary_ratio"]),
        few_shot_examples=int(config["few_shot_examples"]),
        reasoning_level=str(config["reasoning_level"]),
        format_style=str(config["format_style"]),
    )
    answer, usage = call_llm(prompt)
    quality = evaluate_answer(answer)
    shortfall = max(0.0, 0.85 - quality)
    objective = usage.total_tokens + 30000 * (shortfall + shortfall**2)
    return objective, {
        "tokens": usage.total_tokens,
        "quality": quality,
        "quality_floor": 0.85,
        "latency_ms": usage.latency_ms,
    }


optimizer = TokenBayesOptimizer(
    search_space=SearchSpace.token_prompt_space(),
    objective=real_objective,
    config=TokenOptimizationConfig(iterations=40, quality_floor=0.85),
)
result = optimizer.run()
```

## Limites actuelles

- La qualite est simulee dans la demo.
- Le jeu multi-taches est un exemple synthetique, pas encore un benchmark reel.
- Il n'y a pas encore de dashboard interactif.
- Le Gaussian Process est volontairement simple pour eviter les dependances
  lourdes.

## Criteres de passage en production

- remplacer la qualite simulee par une evaluation reelle;
- mesurer les tokens avec le fournisseur IA utilise;
- journaliser le modele, le prompt et la version du benchmark;
- comparer avec une baseline stable;
- definir une qualite minimale par famille de taches;
- verifier que les donnees sensibles ne sont pas exportees dans les rapports.

## Prochaines ameliorations recommandees

1. Ajouter un evaluateur de qualite reel.
2. Ajouter un dashboard Streamlit.
3. Sauvegarder les experiences dans SQLite.
4. Ajouter une baseline grid search.
5. Ajouter des benchmarks metier reels par type de prompt.

## Pourquoi bayesien ?

Une recherche exhaustive teste trop de combinaisons. Une recherche aleatoire
peut gaspiller beaucoup d'essais. L'optimisation bayesienne apprend un modele
approximatif du cout et de la qualite, puis choisit intelligemment les prochains
essais. C'est particulierement utile quand chaque evaluation coute des tokens,
du temps et de l'argent.

## Lien avec les autres projets

Le meme schema peut servir a plusieurs applications:

| Projet | Parametres optimises | Score minimise |
| --- | --- | --- |
| Tokens IA | contexte, RAG, exemples, format | tokens + penalite de qualite |
| Drug discovery | filtres molecule, docking, ADMET | risque - activite - developpabilite |
| Securite defensive | profondeur de scan, seuils, profils de regles | temps/bruit - bonus findings |

La partie commune est l'optimiseur bayesien. La partie specifique est la
fonction objectif: elle transforme une configuration en score metier.

