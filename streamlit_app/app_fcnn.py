import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
from tensorflow.keras.applications.mobilenet import preprocess_input

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="Fish Species Classifier", layout="centered")
st.title("🐟 Fish Species Classification")
st.write("Upload a fish image and the model will predict its species.")

IMG_SIZE = 224  # Must match training

# ---------------- LOAD CLASS NAMES ----------------
@st.cache_resource
def load_class_names():
    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    # If saved as dict → convert to ordered list
    if isinstance(class_names, dict):
        class_names = [class_names[str(i)] for i in range(len(class_names))]

    return class_names

CLASS_NAMES = load_class_names()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    best_model = tf.keras.models.load_model("best_mobilenet.h5")
    return best_model

model = load_model()

# ---------------- IMAGE PREPROCESS FUNCTION ----------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32")

    # ✅ CORRECT preprocessing for MobileNet
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

    # ---------------- PREDICTION ----------------
    predictions = model.predict(input_image)

    # ✅ MODEL ALREADY HAS SOFTMAX → DON'T APPLY AGAIN
    probabilities = predictions[0]

    predicted_index = np.argmax(probabilities)
    confidence = float(probabilities[predicted_index]) * 100

    # ---------------- RESULT ----------------
    st.subheader("✅ Prediction Result")
    st.success(f"**Fish Species:** {CLASS_NAMES[predicted_index]}")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")

    # ---------------- ALL CLASS SCORES ----------------
    st.subheader("📈 Confidence Scores (All Classes)")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {probabilities[i]*100:.2f}%")
