# Multimodal Pneumonia Fusion Detection

A deep learning pipeline that detects pneumonia from chest X-rays by fusing image-based features (via a pretrained ResNet-18) with structured clinical features, and compares performance against image-only and clinical-only baselines.

## Overview

This project trains and evaluates three models on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset:

1. **Image-only** — ResNet-18 embeddings (512-dim) fed into a classifier
2. **Clinical-only** — hand-crafted clinical features (7-dim)
3. **Fusion** — concatenated image + clinical features (519-dim)

Each model is evaluated on accuracy, precision, recall, F1, and ROC-AUC.

## Project Structure
## Setup

### 1. Clone the repo
```bash
git clone https://github.com/DarshanaSethia21/Multimodal-Pneumonia-Fusion-Detection.git
cd Multimodal-Pneumonia-Fusion-Detection
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle API credentials
- Download `kaggle.json` from your [Kaggle account settings](https://www.kaggle.com/settings) (API section).
- Place it at `~/.kaggle/kaggle.json` (on Windows: `$env:USERPROFILE\.kaggle\kaggle.json`).

### 5. Download the dataset
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw
Expand-Archive -Path data/raw/chest-xray-pneumonia.zip -DestinationPath data/raw
```

## Usage

### 1. Extract clinical features
```bash
python src/clinical_features.py
```

### 2. Extract image embeddings
```bash
python src/extract_embeddings.py
```

### 3. Train and evaluate all models
```bash
python main.py --epochs 20
```

Results are saved to `results/metrics/final_results.csv` and comparison plots to `results/figures/`.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Image-only | 0.840 | 0.803 | 0.985 | 0.885 | 0.963 |
| Clinical-only | 0.987 | 0.990 | 0.990 | 0.990 | 0.999 |
| Fusion | 0.973 | 0.965 | 0.992 | 0.979 | 0.998 |

> **Note:** The unusually high clinical-only performance is under investigation for potential data leakage (e.g. feature correlation with label via dataset structure) and validation set size limitations (only 16 images in the original Kaggle `val/` split). Results will be updated after a stratified re-validation.

## Tech Stack
- PyTorch / torchvision (ResNet-18 feature extraction)
- scikit-learn (metrics, preprocessing)
- pandas / numpy
- Streamlit (optional demo interface)

## License
[Add your license here]

## Sample Output

### Model Comparison
![Model Comparison](results/figures/model_comparison.png)

### ROC Curves
![ROC Curves](results/figures/roc_curves.png)