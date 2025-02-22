# Automated Hyperparameter Optimization

## Introduction

Hyperparameter optimization is the process of selecting the best set of hyperparameters to improve a machine learning model's performance. Instead of manually tuning these parameters, we use automated optimization techniques like Grid Search, Randomized Search, Bayesian Optimization, and Genetic Algorithms.

In this project, we compare different hyperparameter tuning methods using a **RandomForest classifier** on the **Diabetes dataset**.

## Dataset
We use the publicly available **Pima Indians Diabetes dataset** from the UCI Machine Learning Repository.

- **Features:** Age, BMI, Glucose Level, Blood Pressure, etc.
- **Target Variable:** Diabetes diagnosis (0 = No, 1 = Yes)

## Methodology
The following steps were followed:

1. **Data Preprocessing:**
   - Replace biologically implausible zero values (e.g., for Glucose, BMI) with the median.
   - Standardize the features using `StandardScaler`.
   - Handle class imbalance using **SMOTE**.

2. **Baseline Model:**
   - Train a simple **RandomForestClassifier**.
   - Evaluate accuracy and ROC AUC score.

3. **Hyperparameter Optimization Techniques:**
   - **Manual Hyperparameter Tuning** (Fixed parameters)
   - **Grid Search** (`GridSearchCV`)
   - **Randomized Search** (`RandomizedSearchCV`)
   - **Bayesian Optimization** (`Hyperopt`)
   - **Sequential Model-Based Optimization** (`Skopt`)
   - **Optuna Optimization** (`Optuna`)
   - **Genetic Algorithm** (`TPOT`)

4. **Comparison of Different Methods:**
   - Evaluate models using Accuracy and ROC AUC scores, storing results for comparison.
   - Compare results to determine the best approach.

## Results
A comparison of different hyperparameter optimization methods:

| Method              | Accuracy | ROC AUC |
|---------------------|----------|---------|
| Baseline Model     | 81.50%   | 81.40%  |
| Grid Search       | 81.50%   | 81.40%  |
| Randomized Search | 78.50%   | 78.46%  |
| Bayesian Opt (Hyperopt) | 81.00% | 81.00%  |
| Sequential (Skopt) | 81.00%   | 80.90%  |
| Optuna            | 79.50%   | 79.40%  |
| TPOT (Genetic Alg) | 80.00%  | 80.00%  |

## Key Takeaways
- **Grid Search** is exhaustive but computationally expensive.
- **Randomized Search** is faster but may not always find the best hyperparameters.
- **Bayesian Optimization (Hyperopt & Skopt)** is more efficient in finding optimal values.
- **Optuna** automates the tuning process efficiently.
- **TPOT (Genetic Algorithm)** automates feature engineering and hyperparameter selection but requires more computational time.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/premnand/Automated-Hyperparameter-Optimisation.git
cd Automated-Hyperparameter-Optimisation
```

2. Install dependencies with:
```bash
pip install numpy pandas scikit-learn hyperopt scikit-optimize optuna tpot imbalanced-learn matplotlib seaborn
```

3. Open the Jupyter Notebook:
```bash
jupyter notebook code.ipynb
```

## Contributing
Feel free to contribute! You can:
- Report issues
- Suggest improvements
- Add new hyperparameter tuning techniques

To contribute, fork the repository and submit a pull request. Ensure to follow coding standards and provide relevant documentation.

---
## Acknowledgments
- **Dataset**: Pima Indians Diabetes Dataset (`diabetes.csv`)

For questions or feedback, contact me at `premnand5657@gmail.com` or open an issue on GitHub.
