# Multimodal Pneumonia Fusion Detection

Detecting pneumonia from chest X-rays by fusing image-based deep features with structured tabular features, and comparing performance against image-only and tabular-only baselines.

## Overview

This project explores a multimodal fusion pipeline for pneumonia classification from chest X-rays, combining a pretrained ResNet-18 image branch with a structured tabular branch. Because the source dataset has no paired clinical/EHR data, the tabular features are **synthetically simulated** (see Data Note below) — this project should be read as an engineering demonstration of the fusion architecture and pipeline, not a clinically validated result.

1. **Image-only** — ResNet-18 embeddings fed into a classifier
2. **Clinical-only** — simulated tabular features fed into a classifier
3. **Fusion** — concatenated image + tabular features (519-dim)

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle, via Mendeley/Guangzhou Women and Children's Medical Center), containing:

- **Train**: 5,216 images (NORMAL / PNEUMONIA)
- **Validation**: 16 images
- **Test**: 624 images

Run `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw` to fetch the dataset (requires a Kaggle API key — see Installation below).

## Data Note — Simulated Clinical Features

The Kaggle dataset contains only images and NORMAL/PNEUMONIA labels — no accompanying patient vitals or lab data. To build and test a fusion architecture, `src/clinical_features.py` generates **synthetic** tabular features (age, sex, smoker status, oxygen saturation, respiratory rate, WBC count, fever temperature) sampled from label-conditioned distributions (e.g. pneumonia-labeled samples are drawn with lower simulated oxygen saturation and higher simulated WBC count).

**This means the "Clinical-only" and "Fusion" results below reflect how well each model can decode a synthetic signal that is directly derived from the label — not real diagnostic performance.** They are included to validate that the fusion pipeline (feature scaling, concatenation, training loop, evaluation) works correctly end-to-end, and are not evidence that clinical data improves real pneumonia detection.

## Methodology

1. **Simulated clinical features** — `src/clinical_features.py` generates 7 synthetic, label-conditioned features per image (see Data Note above), saved to `data/raw/{train,val,test}_clinical.csv`.
2. **Image embedding extraction** — `src/extract_embeddings.py` runs a pretrained ResNet-18 (ImageNet weights) over each image and caches 512-dim embeddings to `data/processed/`.
3. **Feature scaling** — tabular features are scaled with parameters fit on the training split only, to avoid leakage into val/test.
4. **Training** — three separate classifiers (image-only, clinical-only, fusion) are trained for 20 epochs each.
5. **Evaluation** — all three models are scored on the same held-out test set: accuracy, precision, recall, F1, and ROC-AUC. Model size and inference latency are also measured for the fusion model.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Image-only | 0.840 | 0.803 | 0.985 | 0.885 | 0.963 |
| Clinical-only (synthetic) | 0.987 | 0.990 | 0.990 | 0.990 | 0.999 |
| Fusion (synthetic clinical + image) | 0.973 | 0.965 | 0.992 | 0.979 | 0.998 |

**Fusion model size:** 136.9 KB | **Inference:** 1.454 ms

**Image-only is the only model trained on real diagnostic signal.** The clinical-only and fusion numbers are near-ceiling because the tabular features are synthetically generated from the label (see Data Note) — they demonstrate the pipeline works end-to-end, not that clinical data improves diagnosis.

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

- Replace synthetic tabular features with a real paired clinical dataset (e.g. MIMIC-CXR) to make the fusion comparison diagnostically meaningful
- Stratified train/val split to replace the tiny 16-image original validation set
- Confusion matrices and per-model ablations
- SHAP-based explainability for feature contributions

## Disclaimer

This is a research/portfolio project demonstrating a multimodal fusion architecture. The clinical/tabular features are synthetically simulated, not real patient data — see the Data Note above. Predictions are not a medical diagnosis, the model has not been clinically validated, and the reported clinical-only/fusion metrics should not be interpreted as evidence of real diagnostic performance.