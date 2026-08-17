import os
import numpy as np
import pandas as pd


def list_images(split_dir: str) -> pd.DataFrame:
    rows = []
    for label_name, label_value in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        class_dir = os.path.join(split_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                rows.append({
                    "image_path": os.path.join(class_dir, fname),
                    "label": label_value,
                })
    return pd.DataFrame(rows)


def generate_clinical_features(image_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n = len(image_df)
    label = image_df["label"].to_numpy()

    age = rng.normal(loc=45 + label * 8, scale=18).clip(0, 95)
    sex = rng.integers(0, 2, size=n)
    smoker = rng.binomial(1, p=0.15 + 0.10 * label)
    oxygen_saturation = rng.normal(loc=97 - label * 4, scale=2).clip(70, 100)
    respiratory_rate = rng.normal(loc=16 + label * 8, scale=3).clip(8, 45)
    wbc_count = rng.normal(loc=7.5 + label * 4, scale=2).clip(2, 30)
    fever_temp_c = rng.normal(loc=36.8 + label * 1.1, scale=0.5).clip(35, 41)

    out = image_df.copy()
    out["age"] = age.round(1)
    out["sex"] = sex
    out["smoker"] = smoker
    out["oxygen_saturation"] = oxygen_saturation.round(1)
    out["respiratory_rate"] = respiratory_rate.round(1)
    out["wbc_count"] = wbc_count.round(2)
    out["fever_temp_c"] = fever_temp_c.round(1)

    return out


def build_split_csv(dataset_root: str, split: str, out_dir: str = "data/raw") -> pd.DataFrame:
    split_dir = os.path.join(dataset_root, split)
    image_df = list_images(split_dir)

    if len(image_df) == 0:
        raise FileNotFoundError(
            f"No images found under {split_dir}. Check that the Kaggle dataset "
            f"was extracted correctly and dataset_root points at the folder "
            f"containing train/, val/, and test/ subfolders."
        )

    full_df = generate_clinical_features(image_df)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_clinical.csv")
    full_df.to_csv(out_path, index=False)
    print(f"{split}: {len(full_df)} images -> {out_path}")
    return full_df


if __name__ == "__main__":
    DATASET_ROOT = "data/raw/chest_xray/chest_xray"
    for split in ["train", "val", "test"]:
        build_split_csv(DATASET_ROOT, split)

