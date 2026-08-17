import os
import time
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_torch_model(model, path: str) -> float:
    torch.save(model.state_dict(), path)
    return os.path.getsize(path) / 1024


def measure_inference_time(predict_fn, X, n_repeats: int = 20) -> float:
    predict_fn(X)
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        predict_fn(X)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return sum(times) / len(times)


CLINICAL_FEATURE_COLUMNS = [
    "age", "sex", "smoker", "oxygen_saturation",
    "respiratory_rate", "wbc_count", "fever_temp_c",
]


def load_split(split: str, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    clinical_df = pd.read_csv(os.path.join(raw_dir, f"{split}_clinical.csv"))
    embeddings = np.load(os.path.join(processed_dir, f"{split}_embeddings.npy"))
    labels = np.load(os.path.join(processed_dir, f"{split}_labels.npy"))

    clinical_features = clinical_df[CLINICAL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    assert len(clinical_features) == len(embeddings) == len(labels), (
        f"Mismatched lengths for split '{split}': "
        f"clinical={len(clinical_features)}, embeddings={len(embeddings)}, labels={len(labels)}"
    )

    return clinical_features, embeddings, labels