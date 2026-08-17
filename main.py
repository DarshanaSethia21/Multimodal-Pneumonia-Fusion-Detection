import argparse
import json
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils import set_seed, load_split, save_torch_model, measure_inference_time
from src.model import FusionClassifier, build_fusion_input
from src.train import train_classifier, predict_proba
from src.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_metric_comparison, plot_training_curves,
)


def main(epochs: int):
    set_seed(42)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    print("\n=== Step 1: Loading cached features ===")
    clin_train, emb_train, y_train = load_split("train")
    clin_val, emb_val, y_val = load_split("val")
    clin_test, emb_test, y_test = load_split("test")
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    print("\n=== Step 2: Scaling clinical features (fit on train only) ===")
    scaler = StandardScaler()
    clin_train_s = scaler.fit_transform(clin_train)
    clin_val_s = scaler.transform(clin_val)
    clin_test_s = scaler.transform(clin_test)

    fusion_train = build_fusion_input(emb_train, clin_train_s)
    fusion_val = build_fusion_input(emb_val, clin_val_s)
    fusion_test = build_fusion_input(emb_test, clin_test_s)

    experiments = {
        "Image-only": (emb_train, emb_val, emb_test, emb_train.shape[1]),
        "Clinical-only": (clin_train_s, clin_val_s, clin_test_s, clin_train_s.shape[1]),
        "Fusion": (fusion_train, fusion_val, fusion_test, fusion_train.shape[1]),
    }

    all_metrics = []
    roc_data = {}
    trained_models = {}

    for name, (X_tr, X_va, X_te, input_dim) in experiments.items():
        print(f"\n=== Training: {name} (input_dim={input_dim}) ===")
        model = FusionClassifier(input_dim=input_dim)
        model, history = train_classifier(
            model, X_tr, y_train, X_va, y_val, epochs=epochs, print_every=max(epochs // 5, 1)
        )

        test_probs = predict_proba(model, X_te)
        metrics = compute_metrics(y_test, test_probs, model_name=name)
        all_metrics.append(metrics)
        roc_data[name] = (y_test, test_probs)
        trained_models[name] = model

        print(f"[{name}] Accuracy: {metrics['accuracy']:.3f} | "
              f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | "
              f"F1: {metrics['f1']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}")

        safe_name = name.lower().replace("-", "_")
        plot_confusion_matrix(y_test, test_probs, name, f"results/figures/{safe_name}_confusion.png")
        plot_training_curves(history, name, f"results/figures/{safe_name}_training_curve.png")
        save_torch_model(model, f"models/{safe_name}.pt")

    print("\n=== Step: Comparison plots ===")
    plot_roc_curves(roc_data, "results/figures/roc_comparison.png")
    plot_metric_comparison(all_metrics, "results/figures/metric_comparison.png")

    fusion_model = trained_models["Fusion"]
    inference_ms = measure_inference_time(lambda x: predict_proba(fusion_model, x), fusion_test)
    fusion_size_kb = os.path.getsize("models/fusion.pt") / 1024
    print(f"\nFusion model size: {fusion_size_kb:.1f} KB | Inference: {inference_ms:.3f} ms")

    import pandas as pd
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv("results/metrics/final_results.csv", index=False)
    print("\n=== Final Results ===")
    print(results_df.to_string(index=False))

    with open("results/metrics/summary.json", "w") as f:
        json.dump({
            "fusion_size_kb": fusion_size_kb,
            "fusion_inference_ms": inference_ms,
            "clinical_feature_columns": list(clin_train.shape),
        }, f, indent=2)

    print("\nDone. Check results/metrics/final_results.csv and results/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    main(args.epochs)