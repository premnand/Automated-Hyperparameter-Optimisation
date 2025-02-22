# Automated Hyperparameter Optimisation

# Introduction
# Hyperparameter optimization is the process of finding the best set of hyperparameters for a machine learning model. Instead of manually tuning parameters, we use optimization techniques like Grid Search, Random Search, Bayesian Optimization, and Genetic Algorithms to find the best parameters efficiently.

# In this notebook, we will compare different hyperparameter tuning methods using a RandomForest classifier on the Diabetes dataset.

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from hyperopt import fmin, tpe, hp, Trials
from skopt import BayesSearchCV
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
from tpot import TPOTClassifier
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Data Preprocessing
# Load Dataset
df = pd.read_csv("diabetes.csv")
df['Glucose'] = np.where(df['Glucose'] == 0, df['Glucose'].median(), df['Glucose'])

# Dependent and Independent Columns
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Apply Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Baseline Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

baseline_accuracy = accuracy_score(y_test, y_pred)
baseline_roc_auc = roc_auc_score(y_test, y_pred)

print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline ROC AUC Score: {baseline_roc_auc:.4f}")

# Store results
results = {"Baseline": {"Accuracy": baseline_accuracy, "ROC AUC": baseline_roc_auc}}

# Manual Hyperparameter Tuning
model = RandomForestClassifier(n_estimators=200, criterion='entropy',
                               max_features='sqrt', min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Automated Hyperparameter tuning

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train, y_train)
y_pred = grid_search.best_estimator_.predict(X_test)
results["GridSearch"] = {"Accuracy": accuracy_score(y_test, y_pred), "ROC AUC": roc_auc_score(y_test, y_pred)}
print(f"Best Params (Grid Search): {grid_search.best_params_}")

# Randomized Search
param_dist = {
    'n_estimators': np.arange(50, 300, 50),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': np.arange(2, 11, 2)
}
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, scoring='roc_auc')
random_search.fit(X_train, y_train)
y_pred = random_search.best_estimator_.predict(X_test)
results["RandomizedSearch"] = {"Accuracy": accuracy_score(y_test, y_pred), "ROC AUC": roc_auc_score(y_test, y_pred)}
print(f"Best Params (Randomized Search): {random_search.best_params_}")

# Bayesian Optimization (Hyperopt)
def objective(params):
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {'loss': -roc_auc_score(y_test, y_pred), 'status': 'ok'}

space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200, 250]),
    'max_depth': hp.choice('max_depth', [None, 10, 20, 30]),
    'min_samples_split': hp.choice('min_samples_split', [2, 4, 6, 8, 10])
}

trials = Trials()
best_params_idx = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

best_params = {
    'n_estimators': [50, 100, 150, 200, 250][best_params_idx['n_estimators']],
    'max_depth': [None, 10, 20, 30][best_params_idx['max_depth']],
    'min_samples_split': [2, 4, 6, 8, 10][best_params_idx['min_samples_split']]
}

best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
results["Hyperopt"] = {"Accuracy": accuracy_score(y_test, y_pred), "ROC AUC": roc_auc_score(y_test, y_pred)}
print(f"Best Params (Hyperopt): {best_params}")

# Optuna Optimization
def optuna_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, 50),
        'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 2)
    }
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return roc_auc_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(optuna_objective, n_trials=10)
y_pred = RandomForestClassifier(**study.best_params, random_state=42).fit(X_train, y_train).predict(X_test)
results["Optuna"] = {"Accuracy": accuracy_score(y_test, y_pred), "ROC AUC": roc_auc_score(y_test, y_pred)}
print(f"Best Params (Optuna): {study.best_params}")

# Conclusion - Final results
print(results)
