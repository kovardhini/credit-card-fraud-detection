"""
train.py

Applies SMOTE to the training data only, trains three candidate models
(Logistic Regression, Random Forest, XGBoost), compares them by PR-AUC,
and returns the best one along with its test-set probability scores.
"""

import os
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, classification_report

from preprocessing import run_preprocessing

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def apply_smote(X_train, y_train, random_state: int = 42):
    """Balance the training set only. Never call this on test data."""
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"After SMOTE:  {y_train_smote.value_counts().to_dict()}")
    return X_train_smote, y_train_smote


def train_all_models(X_train_smote, y_train_smote, X_test, y_test):
    """Train each candidate model and score it on the untouched test set."""
    results = {}

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_smote, y_train_smote)
    prob_log = log_reg.predict_proba(X_test)[:, 1]
    results["Logistic Regression"] = {
        "model": log_reg,
        "pr_auc": average_precision_score(y_test, prob_log),
        "y_prob": prob_log,
    }

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_smote, y_train_smote)
    prob_rf = rf.predict_proba(X_test)[:, 1]
    results["Random Forest"] = {
        "model": rf,
        "pr_auc": average_precision_score(y_test, prob_rf),
        "y_prob": prob_rf,
    }

    xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb.fit(X_train_smote, y_train_smote)
    prob_xgb = xgb.predict_proba(X_test)[:, 1]
    results["XGBoost"] = {
        "model": xgb,
        "pr_auc": average_precision_score(y_test, prob_xgb),
        "y_prob": prob_xgb,
    }

    print("\nModel comparison (PR-AUC):")
    for name, info in results.items():
        print(f"  {name}: {info['pr_auc']:.4f}")

    return results


def pick_best_model(results: dict):
    """Return the (name, info) pair with the highest PR-AUC."""
    best_name = max(results, key=lambda name: results[name]["pr_auc"])
    print(f"\nBest model: {best_name} (PR-AUC = {results[best_name]['pr_auc']:.4f})")
    return best_name, results[best_name]


def run_training():
    X_train, X_test, y_train, y_test, scaler_amount, scaler_time = run_preprocessing()
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    results = train_all_models(X_train_smote, y_train_smote, X_test, y_test)
    best_name, best_info = pick_best_model(results)

    y_pred = (best_info["y_prob"] >= 0.5).astype(int)
    print(f"\nClassification report for {best_name} at default 0.5 threshold:")
    print(classification_report(y_test, y_pred))

    return best_name, best_info, X_test, y_test, scaler_amount, scaler_time


if __name__ == "__main__":
    run_training()