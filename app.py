import numpy as np
import streamlit as st
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.model import FusionClassifier, build_fusion_input
from src.extract_embeddings import load_frozen_resnet18, IMAGE_TRANSFORM
from src.train import predict_proba
from src.utils import CLINICAL_FEATURE_COLUMNS


st.set_page_config(page_title="Pneumonia Fusion Detector", page_icon="🫁")
st.title("🫁 Multimodal Pneumonia Detector")
st.caption("Fuses a chest X-ray image with clinical values for a combined prediction. "
           "Clinical features in this demo are synthetic — for architecture "
           "demonstration only, not a real diagnostic tool.")


@st.cache_resource
def load_models():
    resnet = load_frozen_resnet18()

    train_clin = pd.read_csv("data/raw/train_clinical.csv")[CLINICAL_FEATURE_COLUMNS]
    scaler = StandardScaler()
    scaler.fit(train_clin.to_numpy(dtype=np.float32))

    fusion_model = FusionClassifier(input_dim=512 + len(CLINICAL_FEATURE_COLUMNS))
    fusion_model.load_state_dict(torch.load("models/fusion.pt"))
    fusion_model.eval()

    return resnet, scaler, fusion_model


try:
    resnet, scaler, fusion_model = load_models()
except FileNotFoundError:
    st.error("No trained model found. Run `python main.py` first (after "
             "clinical_features.py and extract_embeddings.py).")
    st.stop()

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

st.subheader("Clinical values")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 0, 100, 45)
    sex = st.selectbox("Sex", ["Female", "Male"])
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    oxygen_saturation = st.number_input("Oxygen saturation (%)", 70.0, 100.0, 97.0)
with col2:
    respiratory_rate = st.number_input("Respiratory rate (breaths/min)", 8.0, 45.0, 16.0)
    wbc_count = st.number_input("WBC count (x10^9/L)", 2.0, 30.0, 7.5)
    fever_temp_c = st.number_input("Temperature (°C)", 35.0, 41.0, 36.8)

if uploaded_file and st.button("Predict", type="primary"):
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded X-ray", width=300)

    img_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(img_tensor).numpy()

    clinical_raw = np.array([[age, 1 if sex == "Male" else 0, 1 if smoker == "Yes" else 0,
                               oxygen_saturation, respiratory_rate, wbc_count, fever_temp_c]],
                             dtype=np.float32)
    clinical_scaled = scaler.transform(clinical_raw)

    fusion_input = build_fusion_input(embedding, clinical_scaled)
    prob = predict_proba(fusion_model, fusion_input)[0]

    label = "Pneumonia" if prob >= 0.5 else "Normal"
    st.success(f"### Prediction: {label}")
    st.write(f"Predicted probability of pneumonia: **{prob:.1%}**")