# Multimodal Pneumonia Fusion Detection

Detecting pneumonia from chest X-rays by fusing image-based deep features with structured clinical features, and comparing performance against image-only and clinical-only baselines.

## Overview

This project trains and evaluates three models on chest X-ray images to classify pneumonia vs. normal cases. A pretrained ResNet-18 extracts 512-dimensional image embeddings, and a separate set of hand-crafted clinical features (7-dim) is derived per image. Three models are trained and compared:

1. **Image-only** — ResNet-18 embeddings fed into a classifier
2. **Clinical-only** — hand-crafted clinical features
3. **Fusion** — concatenated image + clinical features (519-dim)

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle, via Mendeley/Guangzhou Women and Children's Medical Center), containing:

- **Train**: 5,216 images (NORMAL / PNEUMONIA)
- **Validation**: 16 images
- **Test**: 624 images

Run `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw` to fetch the dataset (requires a Kaggle API key — see Installation below).

## Methodology

1. **Clinical feature extraction** — `src/clinical_features.py` derives 7 structured features per image, saved to `data/raw/{train,val,test}_clinical.csv`.
2. **Image embedding extraction** — `src/extract_embeddings.py` runs a pretrained ResNet-18 (ImageNet weights) over each image and caches 512-dim embeddings to `data/processed/`.
3. **Feature scaling** — clinical features are scaled with parameters fit on the training split only, to avoid leakage into val/test.
4. **Training** — three separate classifiers (image-only, clinical-only, fusion) are trained for 20 epochs each.
5. **Evaluation** — all three models are scored on the same held-out test set: accuracy, precision, recall, F1, and ROC-AUC. Model size and inference latency are also measured for the fusion model.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Image-only | 0.840 | 0.803 | 0.985 | 0.885 | 0.963 |
| Clinical-only | 0.987 | 0.990 | 0.990 | 0.990 | 0.999 |
| Fusion | 0.973 | 0.965 | 0.992 | 0.979 | 0.998 |

**Fusion model size:** 136.9 KB | **Inference:** 1.454 ms

> **Note:** Clinical-only performance is unusually high relative to image-only and fusion, and is under investigation for potential data leakage (e.g. feature correlation with label via dataset structure) and the small validation set size (16 images). Results will be updated after a stratified re-validation.

### Model Comparison

![Model Comparison](results/figures/model_comparison.png)

### ROC Curves

![ROC Curves](results/figures/roc_curves.png)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

### Kaggle API setup

Download `kaggle.json` from your [Kaggle account settings](https://www.kaggle.com/settings) (API section) and place it at `~/.kaggle/kaggle.json` (Windows: `$env:USERPROFILE\.kaggle\kaggle.json`).

## Running the project

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw
python src/clinical_features.py
python src/extract_embeddings.py
python main.py --epochs 20
```

Results are saved to `results/metrics/final_results.csv` and comparison plots to `results/figures/`.

## Streamlit Demo

```bash
streamlit run app.py
```

## Project Structure
## Future Work

- Resolve suspected data leakage in clinical feature extraction
- Stratified train/val split to replace the tiny 16-image original validation set
- Confusion matrices and per-model ablations
- SHAP-based explainability for clinical feature contributions

## Disclaimer

This is a research/portfolio project. Predictions are not a medical diagnosis and the model has not been clinically validated.