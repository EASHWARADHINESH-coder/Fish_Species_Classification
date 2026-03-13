import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
from pathlib import Path
from tensorflow.keras.applications.mobilenet import preprocess_input

# ---------------- PATH SETUP ----------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

CLASS_NAMES_PATH = ROOT_DIR / "class_names.json"
MODEL_PATH = ROOT_DIR / "best_mobilenet.h5"

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="Fish Species Classifier", layout="centered")
st.title("🐟 Fish Species Classification")
st.write("Upload a fish image and the model will predict its species.")

IMG_SIZE = 224

# ---------------- LOAD CLASS NAMES ----------------
@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    if isinstance(class_names, dict):
        class_names = [class_names[str(i)] for i in range(len(class_names))]

    return class_names

CLASS_NAMES = load_class_names()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    best_model = tf.keras.models.load_model(MODEL_PATH)
    return best_model

model = load_model()

# ---------------- IMAGE PREPROCESS FUNCTION ----------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32")
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader("📤 Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Analyzing image...")

    input_image = preprocess_image(image)
    predictions = model.predict(input_image)
    probabilities = predictions[0]

    predicted_index = np.argmax(probabilities)
    confidence = float(probabilities[predicted_index]) * 100

    st.subheader("✅ Prediction Result")
    st.success(f"**Fish Species:** {CLASS_NAMES[predicted_index]}")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")

    st.subheader("📈 Confidence Scores (All Classes)")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {probabilities[i]*100:.2f}%")