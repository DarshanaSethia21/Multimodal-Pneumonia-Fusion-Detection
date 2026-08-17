import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)


def compute_metrics(y_true, y_probs, model_name: str = "Model", threshold: float = 0.5) -> dict:
    y_pred = (y_probs >= threshold).astype(int)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probs) if len(set(y_true)) > 1 else float("nan"),
    }


def plot_confusion_matrix(y_true, y_probs, model_name: str, save_path: str, threshold: float = 0.5):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{model_name}: Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Normal", "Pneumonia"])
    plt.yticks([0, 1], ["Normal", "Pneumonia"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(results: dict, save_path: str):
    plt.figure(figsize=(6, 6))
    for model_name, (y_true, y_probs) in results.items():
        if len(set(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Image-only vs Clinical-only vs Fusion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metric_comparison(metrics_list: list, save_path: str):
    models = [m["model"] for m in metrics_list]
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    plt.figure(figsize=(9, 5))
    for i, m in enumerate(metrics_list):
        values = [m[name] for name in metric_names]
        plt.bar(x + i * width, values, width, label=m["model"])

    plt.xticks(x + width * (len(models) - 1) / 2, metric_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison: Image-only vs Clinical-only vs Fusion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(history: dict, model_name: str, save_path: str):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train loss")
    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(f"{model_name}: Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()