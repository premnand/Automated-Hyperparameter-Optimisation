# Automated Hyperparameter Optimisation

## 📌 Introduction

Hyperparameter optimisation is the process of selecting the best set of hyperparameters to improve a machine learning model's performance. Instead of manually tuning these parameters, we use automated optimisation techniques like Grid Search, Randomized Search, Bayesian Optimisation, and Genetic Algorithms.

In this project, we compare different hyperparameter tuning methods using a **RandomForest classifier** on the **Diabetes dataset**.

## 📊 Dataset
We use the **Pima Indians Diabetes Dataset** (`diabetes.csv`), which consists of multiple medical predictor variables and one target variable (`Outcome`).

- **Features:** Age, BMI, Glucose Level, Blood Pressure, etc.
- **Target Variable:** Diabetes diagnosis (0 = No, 1 = Yes)

## ⚡ Methodology
The following steps were followed:

1. **Data Preprocessing:**
   - Handle missing values by replacing zeros with the median for numerical features.
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
   - Store accuracy and ROC AUC scores for each method.
   - Compare results to determine the best approach.

## 📈 Results
A comparison of different hyperparameter optimization methods:

| Method              | Accuracy | ROC AUC |
|---------------------|----------|---------|
| Baseline Model     | 81.5%      | 81.4%     |
| Grid Search       | 81.5%      | 81.4%     |
| Randomized Search | 78.5%      | 78.46%     |
| Bayesian Opt (Hyperopt) | 81% | 81%     |
| Sequential (Skopt) | 81%      | 80.9%     |
| Optuna            | 79.5%      | 79.4%     |
| TPOT (Genetic Alg) | 80%     | 80%     |

## 🔥 Key Takeaways
- **Grid Search** is exhaustive but computationally expensive.
- **Randomized Search** is faster but may not always find the best hyperparameters.
- **Bayesian Optimization (Hyperopt & Skopt)** is more efficient in finding optimal values.
- **Optuna** automates the tuning process efficiently.
- **TPOT (Genetic Algorithm)** is effective but takes longer to converge.

## 🤝 Contributing
Feel free to contribute! You can:
- Report issues
- Suggest improvements
- Add new hyperparameter tuning techniques

To contribute, fork the repository and submit a pull request.

## 📜 License
This project is licensed under the **MIT License**.

---

🔗 **Author:** Prem Nand
📧 **Contact:** premnand5657@gmail.com 
📍 **GitHub:** [Your GitHub Profile](https://github.com/premnand)
