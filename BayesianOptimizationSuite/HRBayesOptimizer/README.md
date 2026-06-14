# HR Bayes Optimizer

Systeme d'optimisation bayesienne pour choisir des reglages RH sous contraintes
de cout, qualite d'embauche, retention, equite et bien-etre.

Ce projet contient un moteur historique plus avance que `BayesCore`, notamment
pour les contraintes et le bruit. L'architecture cible du portefeuille est de
conserver les idees metier RH ici, puis de remonter progressivement les briques
generiques utiles dans `C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\BayesCore`.

Documentation commune:

- [Vue d'ensemble des projets](..\BAYESIAN_OPTIMIZATION_PROJECTS.md)
- [Architecture BayesCore](..\BayesCore\ARCHITECTURE.md)
- [Usage professionnel BayesCore](..\BayesCore\PROFESSIONAL_USAGE.md)

## Idee

L'outil cherche automatiquement une politique RH performante:

- budget de sourcing;
- nombre d'entretiens;
- poids donne aux assessments;
- heures d'onboarding;
- jours de remote par semaine;
- bonus de cooptation;
- politique de screening.

Il minimise une fonction objectif:

```text
objectif = cout_RH + penalite_delai + penalites_si_contraintes_non_respectees
```

Les contraintes par defaut sont:

- qualite d'embauche >= 0.76;
- retention >= 0.72;
- bien-etre >= 0.68;
- fairness gap <= 0.10.

## Fonctionnement

1. Le projet definit un espace de recherche RH.
2. Il teste quelques politiques aleatoires pour collecter des observations.
3. Il entraine un petit Gaussian Process qui approxime la fonction objectif.
4. Il utilise Expected Improvement pour choisir la prochaine politique a tester.
5. Il garde la meilleure politique faisable.
6. Il compare cette politique a une baseline RH plus conservative.
7. Il exporte un historique CSV, un JSON complet et deux rapports.

Le simulateur inclus sert de demo reproductible. Dans un vrai systeme, remplace
`HRPolicyObjective` par une fonction qui lit des KPI reels ou lance une
simulation metier.

## Architecture transferable

Le code est separe en deux couches:

- `src/hr_bayes_optimizer/core.py`: moteur generique reutilisable dans n'importe
  quel projet;
- `src/hr_bayes_optimizer/optimizer.py`: adaptation metier RH, avec metriques,
  contraintes, baseline et rapports RH;
- `src/hr_bayes_optimizer/cli.py`: CLI orientee RH.

Le coeur generique contient:

- `Parameter` et `SearchSpace` pour definir les variables a optimiser;
- `BayesianOptimizer` pour lancer l'optimisation;
- `BayesianOptimizationConfig` pour regler iterations, bruit, repetitions et
  exploration;
- `Constraint` pour definir des contraintes metier modelisees separement;
- `GaussianProcessRegressor` et `expected_improvement` pour le modele bayesien.

Quand des contraintes sont fournies, l'acquisition devient:

```text
score = ExpectedImprovement(objectif) * ProbabilityOfFeasibility(contraintes)
```

Cela evite de choisir un candidat prometteur sur le cout mais probablement
non conforme aux contraintes metier.

Pour transferer le moteur vers un autre domaine, garde `core.py` et remplace
seulement:

- l'espace de recherche;
- la fonction objectif;
- les contraintes de faisabilite si necessaire;
- les rapports/metriques metier.

Exemple minimal hors RH:

```python
from hr_bayes_optimizer.core import (
    BayesianOptimizationConfig,
    BayesianOptimizer,
    Constraint,
    Parameter,
    SearchSpace,
)


def objective(config):
    learning_rate = float(config["learning_rate"])
    batch_size = float(config["batch_size"])
    loss = (learning_rate - 0.03) ** 2 + (batch_size - 64) ** 2 / 10000
    return loss, {"loss": loss}


space = SearchSpace(
    parameters=(
        Parameter("learning_rate", "float", low=0.001, high=0.1),
        Parameter("batch_size", "int", low=16, high=128),
    )
)

optimizer = BayesianOptimizer(
    search_space=space,
    objective=objective,
    config=BayesianOptimizationConfig(iterations=30, observation_repeats=2),
    constraints=(
        Constraint("loss", "max", 0.01),
    ),
)

result = optimizer.run()
print(result.best.config, result.best.objective)
```

Pour une contrainte minimum, utilise `Constraint("score", "min", 0.8)`.
Pour une contrainte maximum, utilise `Constraint("latency_ms", "max", 250)`.

Si ton domaine a une logique de faisabilite plus complexe, cree une sous-classe
de `BayesianOptimizer` et surcharge `_is_feasible`. Si tu veux des champs metier
types, surcharge aussi `_make_observation`, comme le fait `HRBayesOptimizer`.

## Utilisation rapide

Objectif concret: trouver une politique RH moins couteuse qui respecte quand
meme les seuils de qualite, retention, equite et bien-etre.

Depuis PowerShell:

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HRBayesOptimizer
$env:PYTHONPATH = "src"
python -m hr_bayes_optimizer.cli --iterations 35
```

Si `python` n'est pas disponible dans le PATH, utilise le Python embarque par
Codex:

```powershell
cd C:\Users\user\Documents\CodeMLPython\BayesianOptimizationSuite\HRBayesOptimizer
$env:PYTHONPATH = "src"
C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m hr_bayes_optimizer.cli --iterations 35
```

Exemple avec des contraintes plus strictes:

```powershell
python -m hr_bayes_optimizer.cli `
  --iterations 60 `
  --quality-floor 0.80 `
  --retention-floor 0.75 `
  --wellbeing-floor 0.70 `
  --max-fairness-gap 0.08
```

Exemple avec gestion du bruit:

```powershell
python -m hr_bayes_optimizer.cli `
  --iterations 60 `
  --observation-repeats 3 `
  --metric-noise 0.015 `
  --cost-noise-eur 150 `
  --gp-noise 0.0001 `
  --exploration 0.02
```

Dans cet exemple:

- chaque politique est evaluee 3 fois puis moyennee;
- `metric-noise` simule du bruit sur qualite, retention, equite et bien-etre;
- `cost-noise-eur` simule du bruit sur les couts;
- `gp-noise` evite au Gaussian Process de sur-apprendre les observations;
- `exploration` pousse l'acquisition a tester des zones encore incertaines.

Pour comparer avec l'ancienne acquisition non contrainte:

```powershell
python -m hr_bayes_optimizer.cli `
  --iterations 60 `
  --unconstrained-acquisition
```

Exemple avec noms de fichiers personnalises:

```powershell
python -m hr_bayes_optimizer.cli `
  --iterations 50 `
  --json-output resultats_rh.json `
  --csv-output historique_rh.csv `
  --report-output rapport_rh.md `
  --html-output rapport_rh.html
```

Reprendre une optimisation existante:

```powershell
python -m hr_bayes_optimizer.cli `
  --resume-from hr_optimization_result.json `
  --iterations 20
```

Dans ce mode, `--iterations` indique le nombre de nouvelles evaluations a
ajouter a l'historique existant. La baseline et les anciennes observations sont
conservees dans les nouveaux exports.

## Comment lire le resultat

La CLI affiche une meilleure politique, par exemple:

```text
Best HR policy
  cost_eur: 9404.46
  quality: 0.8026
  retention: 0.8628
  fairness_gap: 0.0234
  time_to_hire_days: 36.45
  wellbeing: 0.9062
```

Interpretation:

- `cost_eur`: cout estime de la politique RH;
- `quality`: qualite d'embauche estimee, doit rester au-dessus du seuil;
- `retention`: score de retention estime;
- `fairness_gap`: ecart d'equite, plus bas est meilleur;
- `time_to_hire_days`: delai d'embauche estime;
- `wellbeing`: score de bien-etre estime;
- `objective_std`: incertitude observee si `--observation-repeats` est superieur a 1;
- `objective`: score interne optimise, plus bas est meilleur.

La configuration finale indique les leviers RH proposes:

```json
{
  "sourcing_budget_eur": 3310,
  "interview_rounds": 3,
  "assessment_weight": 0.65,
  "onboarding_hours": 55,
  "remote_days_per_week": 3,
  "referral_bonus_eur": 963,
  "screening_policy": "balanced"
}
```

Dans cet exemple, la politique proposee signifie:

- investir environ 3310 EUR en sourcing;
- faire 3 tours d'entretien;
- donner un poids important, mais pas exclusif, aux assessments;
- prevoir 55 heures d'onboarding;
- autoriser 3 jours de remote par semaine;
- proposer un bonus de cooptation d'environ 963 EUR;
- utiliser une politique de screening equilibree.

## Lancer

Depuis ce dossier:

```bash
PYTHONPATH=src python -m hr_bayes_optimizer.cli --iterations 35
```

Sous PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m hr_bayes_optimizer.cli --iterations 35
```

Avec installation editable:

```bash
pip install -e .
hr-bayes-optimize --iterations 35
```

Sorties:

- `hr_optimization_result.json`: meilleure politique et historique complet;
- `hr_optimization_history.csv`: historique exploitable dans Excel ou Power BI;
- `hr_optimization_report.md`: rapport lisible;
- `hr_optimization_report.html`: rapport visuel avec graphique et front de Pareto.

## Exemple d'integration avec de vraies donnees

```python
def real_hr_objective(config):
    metrics = evaluate_policy_with_hr_data(config)
    shortfall = max(0.0, 0.76 - metrics["quality"])
    fairness_excess = max(0.0, metrics["fairness_gap"] - 0.10)
    objective = metrics["cost_eur"] + metrics["time_to_hire_days"] * 110 + 25000 * (shortfall + fairness_excess)
    return objective, metrics
```

Si les donnees reelles sont bruitees, retourne plusieurs mesures via
`--observation-repeats`, ou ajoute `objective_std` dans les metadonnees:

```python
def real_hr_objective(config):
    metrics = evaluate_policy_with_hr_data(config)
    objective = compute_objective(metrics)
    return objective, {
        **metrics,
        "objective_std": metrics.get("objective_std", 0.0),
    }
```

Le Gaussian Process utilise cet ecart-type pour donner moins de poids aux
observations incertaines. L'optimiseur compare aussi les candidats avec le
meilleur point predit par le modele, pas seulement avec la meilleure mesure
brute, ce qui le rend moins sensible a un point chanceux.

## Tests

```bash
python scripts/run_tests.py
```

## Prudence RH

Ce projet optimise des politiques et des budgets, pas des personnes. Pour un
usage reel, garde une validation humaine, audite les biais, documente les
features employees/candidats utilisees, et evite toute decision automatique
individuelle sans controle legal et ethique.

