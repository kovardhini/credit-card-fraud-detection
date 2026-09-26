import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import run_training

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(PROJECT_ROOT, "graphs")


def ensure_graphs_folder():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    print(f"Saving graphs to: {GRAPHS_DIR}")


def save_pr_auc_comparison(results):
    names = list(results.keys())
    scores = [results[n]["pr_auc"] for n in names]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, scores, color=["#888", "#4C9AFF", "#2ECC71"])
    plt.ylabel("PR-AUC")
    plt.title("Model Comparison: PR-AUC (higher is better)")
    plt.ylim(0, 1)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.4f}",
                  ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "pr_auc_comparison.png"), dpi=150)
    plt.close()
    print("Saved pr_auc_comparison.png")


def save_pr_curves(results, y_test):
    plt.figure(figsize=(7, 6))
    for name, info in results.items():
        precision, recall, _ = precision_recall_curve(y_test, info["y_prob"])
        plt.plot(recall, precision, label=f"{name} (PR-AUC={info['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "precision_recall_curves.png"), dpi=150)
    plt.close()
    print("Saved precision_recall_curves.png")


def save_roc_curves(results, y_test):
    plt.figure(figsize=(7, 6))
    for name, info in results.items():
        fpr, tpr, _ = roc_curve(y_test, info["y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (secondary metric)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    print("Saved roc_curves.png")


def save_confusion_matrices(results, y_test, threshold=0.5):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, info) in zip(axes, results.items()):
        y_pred = (info["y_prob"] >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_title(f"{name}\n(threshold={threshold})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "confusion_matrices_comparison.png"), dpi=150)
    plt.close()
    print("Saved confusion_matrices_comparison.png")


def save_best_model_confusion_matrix(y_test, y_prob, best_threshold, model_name):
    y_pred = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title(f"{model_name} - Final Deployed Threshold ({best_threshold:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "final_model_confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved final_model_confusion_matrix.png")


def main():
    ensure_graphs_folder()

    best_name, best_info, X_test, y_test, scaler_amount, scaler_time = run_training()

    from preprocessing import run_preprocessing
    from train import apply_smote, train_all_models

    X_train, X_test, y_train, y_test, _, _ = run_preprocessing()
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    results = train_all_models(X_train_smote, y_train_smote, X_test, y_test)

    save_pr_auc_comparison(results)
    save_pr_curves(results, y_test)
    save_roc_curves(results, y_test)
    save_confusion_matrices(results, y_test, threshold=0.5)

    import json
    with open(os.path.join(PROJECT_ROOT, "threshold.json")) as f:
        deployed_threshold = json.load(f)["threshold"]

    xgb_prob = results["XGBoost"]["y_prob"]
    save_best_model_confusion_matrix(y_test, xgb_prob, deployed_threshold, "XGBoost")

    print("\nAll graphs saved successfully in the 'graphs/' folder.")


if __name__ == "__main__":
    main()
