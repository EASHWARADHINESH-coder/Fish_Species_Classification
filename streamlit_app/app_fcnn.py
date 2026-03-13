import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet import preprocess_input

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

CLASS_NAMES_PATH = ROOT_DIR / "class_names.json"
MODEL_PATH = ROOT_DIR / "best_mobilenet.keras"

st.set_page_config(page_title="Fish Species Classifier", layout="centered")
st.title("🐟 Fish Species Classification")
st.write("Upload a fish image and the model will predict its species.")

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50


@st.cache_resource
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        st.error(f"class_names.json not found at: {CLASS_NAMES_PATH}")
        st.stop()

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if isinstance(class_names, dict):
        class_names = [class_names[str(i)] for i in range(len(class_names))]

    return class_names


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    try:
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    return loaded_model


CLASS_NAMES = load_class_names()
model = load_model()


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32")
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


uploaded_file = st.file_uploader("📤 Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Analyzing image...")

    input_image = preprocess_image(image)
    predictions = model.predict(input_image, verbose=0)
    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    st.subheader("✅ Prediction Result")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ The model is not very confident about this image.")
    else:
        st.success(f"**Fish Species:** {CLASS_NAMES[predicted_index]}")

    st.write(f"📊 Confidence: **{confidence * 100:.2f}%**")

    st.subheader("🏆 Top 3 Predictions")
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    for rank, idx in enumerate(top3_indices, start=1):
        st.write(f"{rank}. {CLASS_NAMES[idx]} — {probabilities[idx] * 100:.2f}%")

    st.subheader("📈 Confidence Scores (All Classes)")
    chart_data = {
        class_name: float(probabilities[i])
        for i, class_name in enumerate(CLASS_NAMES)
    }
    st.bar_chart(chart_data)