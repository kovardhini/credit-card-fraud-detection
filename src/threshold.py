"""
threshold.py

Finds the F1-optimal decision threshold for a trained model's probability
scores, instead of relying on the default 0.5 cutoff.

This replaces the threshold-finding cells in fraud_detection.ipynb.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report


def find_best_threshold(y_test, y_prob):
    """Return the threshold that maximizes F1 score, plus the precision/recall there."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])

    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Precision at this threshold: {precision[best_idx]:.4f}")
    print(f"Recall at this threshold: {recall[best_idx]:.4f}")
    print(f"F1 score at this threshold: {f1_scores[best_idx]:.4f}")

    return best_threshold, precision[best_idx], recall[best_idx], f1_scores[best_idx]


def evaluate_at_threshold(y_test, y_prob, threshold: float, label: str = ""):
    """Print the confusion matrix and classification report at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\nConfusion matrix{' (' + label + ')' if label else ''} at threshold {threshold:.4f}:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return y_pred


if __name__ == "__main__":
    # Quick standalone demo using train.py's pipeline
    from train import run_training

    best_name, best_info, X_test, y_test, scaler_amount, scaler_time = run_training()
    y_prob = best_info["y_prob"]

    best_threshold, precision, recall, f1 = find_best_threshold(y_test, y_prob)
    evaluate_at_threshold(y_test, y_prob, best_threshold, label=best_name)