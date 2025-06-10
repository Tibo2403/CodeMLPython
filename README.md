# Medical Machine Learning Notebooks

A collection of Jupyter notebooks demonstrating common machine‑learning workflows on several open health datasets.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [Breast_cancer_projects.ipynb](Breast_cancer_projects.ipynb) | Classification of the UCI Breast Cancer Wisconsin dataset using KNN, SVM and cross validation. |
| [DNA+CLASSIFICATION.ipynb](DNA+CLASSIFICATION.ipynb) | Classifies promoter gene sequences from the UCI Molecular Biology dataset. Demonstrates preprocessing DNA sequences and evaluating models such as k‑nearest neighbors and Gaussian processes. |
| [Diabetes_Udemy+(1).ipynb](Diabetes_Udemy+(1).ipynb) | Uses the Pima Indians Diabetes dataset (`diabetes.csv`) to build a neural network. Includes data standardization, handling missing values and evaluating model performance. |
| [Heart_disease_project.ipynb](Heart_disease_project.ipynb) | Exploratory data analysis and neural network model on the Cleveland heart disease dataset (`Ch3.ClevelandData.xlsx`). |
| [Heart+Disease+Prediction+with+Neural+Networks.ipynb](Heart+Disease+Prediction+with+Neural+Networks.ipynb) | Loads the processed Cleveland dataset from UCI, converts the labels for multi‑class classification and trains a neural network with Keras. |

## Datasets

| File | Source |
|------|-------|
| [diabetes.csv](diabetes.csv) | Pima Indians Diabetes dataset |
| [Ch3.ClevelandData.xlsx](Ch3.ClevelandData.xlsx) | Cleveland heart disease dataset |

Other notebooks download their data directly from the UCI Machine Learning Repository and do not require local files.

## Running the notebooks locally

1. Create a Python environment (Python 3.8 or newer is recommended).
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib keras jupyter
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open any `.ipynb` file from the interface to run the code. You can also upload the notebooks to [Google Colab](https://colab.research.google.com/) for execution in the cloud.

Datasets included in this repository (`diabetes.csv` and `Ch3.ClevelandData.xlsx`) are ready to use. Notebooks that reference online sources will automatically download the data when run.


## License

This project is licensed under the [MIT License](LICENSE).
