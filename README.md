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

1. Create a Python environment with Python 3.10 or newer.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open any `.ipynb` file and run it.

## Maintenance

Clear notebook outputs before committing:

```bash
scripts/clear_outputs.sh
```

Run the lightweight checks:

```bash
pytest
```

## License

This project is licensed under the [MIT License](LICENSE).
