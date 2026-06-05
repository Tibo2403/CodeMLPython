# Medical Machine Learning Notebooks

A collection of Jupyter notebooks demonstrating common machine-learning workflows on open health datasets.

## Notebooks

| Notebook | Description |
| --- | --- |
| [Breast_cancer_projects.ipynb](Breast_cancer_projects.ipynb) | Classification of the UCI Breast Cancer Wisconsin dataset using KNN, SVM, and cross validation. |
| [DNA+CLASSIFICATION.ipynb](DNA+CLASSIFICATION.ipynb) | Classifies promoter gene sequences from the UCI Molecular Biology dataset. Demonstrates preprocessing DNA sequences and evaluating models such as k-nearest neighbors and Gaussian processes. |
| [Diabetes_Udemy.ipynb](Diabetes_Udemy.ipynb) | Uses the Pima Indians Diabetes dataset (`diabetes.csv`) to build a neural network. Includes data standardization, missing-value handling, and model evaluation. |
| [Heart_disease_project.ipynb](Heart_disease_project.ipynb) | Exploratory data analysis and neural network modeling on the Cleveland heart disease dataset (`Ch3.ClevelandData.csv`). |
| [Heart+Disease+Prediction+with+Neural+Networks.ipynb](Heart+Disease+Prediction+with+Neural+Networks.ipynb) | Loads the processed Cleveland dataset from UCI, converts labels for multi-class classification, and trains a neural network with Keras. |

## Datasets

| File | Source |
| --- | --- |
| [diabetes.csv](diabetes.csv) | Pima Indians Diabetes dataset |
| [Ch3.ClevelandData.csv](Ch3.ClevelandData.csv) | Cleveland heart disease dataset |

Other notebooks download their data directly from the UCI Machine Learning Repository.

## Running Locally

1. Create a Python environment with Python 3.11 or newer.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Optional: install the RDKit-enabled drug discovery environment:

   ```bash
   pip install -r requirements-drug-discovery.txt
   ```

4. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open any `.ipynb` file and run it.

## Statistics Calculator

The repository also includes a small automatic calculator for common statistics
exercises from the course formula sheet:

```bash
python statistics_calculator.py describe 12 15 18 20
python statistics_calculator.py normal --mean 100 --std 15 --lower 85 --upper 115
python statistics_calculator.py ci-mean 12 15 18 20 --confidence 0.95
python statistics_calculator.py ci-proportion --successes 45 --n 100 --confidence 0.95
python statistics_calculator.py ci-variance --sample-variance 4 --n 20
python statistics_calculator.py ci-two-means --case equal-variance --mean1 12 --mean2 10 --std1 2 --std2 3 --n1 15 --n2 12
python statistics_calculator.py ci-two-proportions --successes1 55 --n1 100 --successes2 45 --n2 100
python statistics_calculator.py test-mean 12 15 18 20 --mu0 15
python statistics_calculator.py test-proportion --successes 55 --n 100 --p0 0.5
python statistics_calculator.py test-two-means --case large --mean1 12 --mean2 10 --std1 2 --std2 3 --n1 40 --n2 45
python statistics_calculator.py test-variance --sample-variance 4 --n 20 --variance0 4
python statistics_calculator.py test-variances --variance1 9 --variance2 4 --n1 12 --n2 10
python statistics_calculator.py test-two-proportions --successes1 55 --n1 100 --successes2 45 --n2 100
python statistics_calculator.py chi-square-gof --observed 25 25 50 --probabilities 0.25 0.25 0.5
python statistics_calculator.py chi-square-independence --rows 10,20 20,40
python statistics_calculator.py regression --x 1 2 3 4 --y 2 3 5 4
python statistics_calculator.py binomial --n 10 --k 3 --p 0.2
python statistics_calculator.py poisson --lambda 2 --k 3
```

Covered cases include confidence intervals and hypothesis tests for one mean,
one proportion, one variance, two means, paired means, two proportions, two
variances, chi-square goodness-of-fit, chi-square independence, and simple
linear regression inference.

It can also be imported in a notebook:

```python
from statistics_calculator import descriptive_stats, confidence_interval_mean, test_mean

descriptive_stats([12, 15, 18, 20])
confidence_interval_mean([12, 15, 18, 20], confidence=0.95)
test_mean([12, 15, 18, 20], hypothesized_mean=15)
```

## AI Drug Discovery Method

`ai_drug_discovery.py` provides an automatable method for finding new candidate
molecules for drug research:

1. collect labelled molecules in a CSV with `smiles` and `activity` columns;
2. transform SMILES strings into molecular descriptors;
3. train a QSAR-like activity model with scikit-learn;
4. generate new candidates by adding or substituting small chemical fragments;
5. score candidates by predicted activity, Lipinski filters, Veber filters, and drug-likeness;
6. export the ranked molecules for chemistry review and lab validation.

Example:

```bash
python ai_drug_discovery.py --training-csv examples/molecules.csv --seed "CCO" "c1ccccc1O" --top-n 10 --output candidate_molecules.csv
```

Batch seed CSV, model metrics, and custom multi-objective weights are also
supported:

```bash
python ai_drug_discovery.py --training-csv examples/molecules.csv --seed-csv examples/seeds.csv --evaluate --random-state 42 --activity-weight 0.70 --drug-likeness-weight 0.30 --top-n 10 --output candidate_molecules.csv
```

To also inspect molecules that failed prioritization filters:

```bash
python ai_drug_discovery.py --training-csv examples/molecules.csv --seed-csv examples/seeds.csv --evaluate --random-state 42 --activity-weight 0.70 --drug-likeness-weight 0.30 --top-n 10 --output candidate_molecules.csv --rejected-output rejected_molecules.csv --metrics-output metrics.json --report-output drug_discovery_report.html
```

`metrics.json` records run parameters and model metrics for reproducibility.
`drug_discovery_report.html` provides a standalone summary with top candidates,
rejected molecules, filters, scores, and model metrics.

When RDKit is installed, the method automatically uses canonical SMILES, Morgan
fingerprints, molecular weight, LogP, TPSA, hydrogen-bond donor/acceptor counts,
rotatable bonds, ring counts, Lipinski rule-of-five violations, and Veber oral
bioavailability filters. Without RDKit, it falls back to lightweight descriptors
so the example remains runnable in simple Python environments.

The exported CSV includes predicted activity, final ranking score, Lipinski and
Veber violation counts, molecular weight, LogP, TPSA, hydrogen-bond
donor/acceptor counts, rotatable bonds, ring count, and heavy atom count.

The implementation is intentionally lightweight so it can run in this repository
without specialist chemistry dependencies. For real medicinal chemistry work,
train on validated assay data, add ADMET and toxicity models, and require expert
review before synthesis or biological testing.

## Maintenance

Clear notebook outputs before committing:

```bash
scripts/clear_outputs.sh
```

Run the lightweight checks:

```bash
python scripts/run_tests.py
```

If you have pytest installed, you can also run:

```bash
python -m pytest
```

Audit Python dependencies for known vulnerabilities:

```bash
python -m pip_audit -r requirements.txt --progress-spinner off
```

## License

This project is licensed under the [MIT License](LICENSE).
