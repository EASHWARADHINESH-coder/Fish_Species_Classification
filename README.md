# 🐟 Fish Species Classification Web App

A Deep Learning powered web application that identifies fish species from images using a **MobileNet CNN model** and an interactive **Streamlit interface**.

---

## 📌 Project Overview

This project uses **Transfer Learning** with MobileNet to perform **multi-class image classification** of fish species. The trained model is deployed using Streamlit, allowing users to upload fish images and get real-time predictions.

---

## 🧠 Deep Learning Model

| Feature | Details |
|--------|---------|
| Model Type | Convolutional Neural Network (CNN) |
| Base Model | MobileNet (Pretrained on ImageNet) |
| Technique | Transfer Learning |
| Input Size | 224 x 224 |
| Output | Multi-class classification |
| Framework | TensorFlow / Keras |

---

## 🚀 Application Features

✅ Upload fish images  
✅ Real-time species prediction  
✅ Confidence score display  
✅ Shows probabilities for all classes  
✅ Fast and interactive UI  

---

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app_fcnn.py
