# Security Bayes Optimizer

Application defensive derivee du meme principe que `TokenBayesOptimizer`.

Le moteur bayesien vient maintenant du projet partage
`C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore`. `SecurityBayesOptimizer` ne garde que la
partie metier securite:

- espace de recherche specialise pour le scan statique;
- fonction objectif securite;
- conversion des observations en findings;
- front de Pareto securite;
- exports JSON, CSV et HTML.

Mais l'objectif change: au lieu de reduire les tokens, l'outil optimise une
configuration de scan statique local pour trouver plus de failles potentielles
avec moins de bruit et un temps raisonnable.

## Important

Cet outil est defensif uniquement:

- scan local de fichiers;
- pas d'exploitation;
- pas de scan reseau;
- pas de brute force;
- pas de modification de cible.

Utilise-le uniquement sur du code que tu possedes ou que tu es autorise a
auditer. Les findings sont des signaux de triage, pas des preuves definitives.

## Ce que le scanner detecte

Regles incluses:

- secrets hardcodes potentiels;
- `eval(...)` et `exec(...)`;
- `subprocess(..., shell=True)`;
- mode debug active;
- hash faibles `md5` / `sha1`;
- verification TLS desactivee;
- `yaml.load(...)`;
- SQL construit par concatenation;
- affectation `innerHTML`.

## Parametres optimises

| Parametre | Role |
| --- | --- |
| `max_files` | Nombre maximal de fichiers analyses |
| `max_file_kb` | Taille maximale d'un fichier scanne |
| `entropy_threshold` | Seuil d'entropie pour detecter les secrets |
| `include_tests` | Inclure ou exclure les fichiers de test |
| `rule_profile` | `fast`, `balanced`, ou `deep` |
| `severity_floor` | Severite minimale: `low`, `medium`, `high` |

## Lancer la demo

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\SecurityBayesOptimizer
$env:PYTHONPATH="C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore\src;C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\SecurityBayesOptimizer\src"
python -m security_bayes_optimizer.cli --target examples\demo_project --iterations 20
```

## Lancer sur un projet local autorise

```powershell
python -m security_bayes_optimizer.cli `
  --target C:\Users\user\Documents\MonProjet `
  --iterations 30 `
  --acquisition lower_confidence_bound `
  --exploration-weight 1.6
```

Si tu n'installes pas les packages en mode editable, garde le `PYTHONPATH` avec
les deux dossiers `BayesCore\src` et `SecurityBayesOptimizer\src`.

## Sorties

- `security_optimization_result.json`: resultats complets et findings.
- `security_optimization_history.csv`: historique des configurations testees.
- `security_optimization_report.html`: rapport visuel.

## Lire le rapport

Le rapport HTML sert a trier les resultats, pas a prouver automatiquement une
faille. Les colonnes importantes sont:

| Colonne | Interpretation |
| --- | --- |
| `Findings` | Nombre total de signaux detectes |
| `High` | Signaux classes en severite haute |
| `Precision` | Proxy interne pour limiter le bruit |
| `Elapsed` | Temps de scan mesure |
| `Objective` | Score minimise par l'optimiseur |

Une bonne configuration n'est pas seulement celle qui trouve le plus de signaux.
Elle doit aussi rester rapide et produire un bruit acceptable.

## Options CLI

| Option | Defaut | Role |
| --- | ---: | --- |
| `--target` | requis | Dossier local autorise a scanner |
| `--iterations` | `24` | Nombre de configurations testees |
| `--initial-points` | `8` | Points aleatoires avant le modele bayesien |
| `--candidate-pool` | `300` | Candidats evalues par acquisition a chaque iteration |
| `--random-state` | `42` | Seed reproductible |
| `--acquisition` | `expected_improvement` | Strategie bayesienne |
| `--exploration-weight` | `1.25` | Exploration pour `lower_confidence_bound` |
| `--json-output` | `security_optimization_result.json` | Export JSON |
| `--csv-output` | `security_optimization_history.csv` | Export CSV |
| `--html-output` | `security_optimization_report.html` | Rapport HTML |

## Fonction objectif

Le score cherche a:

- maximiser les findings;
- donner plus de valeur aux findings `high`;
- favoriser une meilleure precision proxy;
- penaliser les scans trop lents.

Le score est minimise:

```text
objective = cout_temps - bonus_findings - bonus_high - bonus_precision
```

## Role du noyau bayesien

Le projet securite fournit au noyau:

1. une configuration candidate, par exemple `rule_profile=deep` et
   `severity_floor=medium`;
2. un scan local defensif;
3. un score numerique a minimiser;
4. des metadonnees exploitables dans le rapport.

Le noyau `BayesCore` decide ensuite quelle configuration tester. Il n'a aucune
connaissance des failles, des fichiers ou des regles SAST. Cette separation rend
le meme moteur reutilisable pour les tokens ou la drug discovery.

Voir aussi:

- [Vue d'ensemble des projets](..\BAYESIAN_OPTIMIZATION_PROJECTS.md)
- [Architecture BayesCore](..\BayesCore\ARCHITECTURE.md)
- [Usage professionnel BayesCore](..\BayesCore\PROFESSIONAL_USAGE.md)

## Workflow recommande

1. Lancer un premier run court sur le projet autorise.
2. Ouvrir le rapport HTML et verifier les findings high.
3. Relancer avec plus d'iterations si les resultats sont instables.
4. Comparer la meilleure configuration avec une configuration conservative.
5. Corriger ou confirmer manuellement les findings.
6. Conserver le JSON comme trace reproductible du run.

## Limites

- Les regles sont statiques et heuristiques.
- Les findings peuvent contenir des faux positifs.
- L'outil ne remplace pas une revue de code securite.
- Il ne teste pas l'exploitation des failles.
- Il ne scanne pas les services reseau.

## Structure

```text
SecurityBayesOptimizer/
  README.md
  pyproject.toml
  examples/
    demo_project/
  scripts/
    run_tests.py
  src/
    security_bayes_optimizer/
      core.py
      scanner.py
      cli.py
  tests/
    test_security_optimizer.py
```

## Tests

```powershell
python scripts\run_tests.py
```

Le runner ajoute automatiquement `BayesCore\src` au chemin Python local.

## Prochaines ameliorations

1. Ajouter un export SARIF pour GitHub Code Scanning.
2. Ajouter plus de langages et de regles.
3. Ajouter une interface Streamlit.
4. Ajouter une baseline random search.
5. Ajouter une verification par tests unitaires ou SAST externe.
6. Ajouter une configuration CI pour lancer un scan defensif sur un dossier
   autorise.

