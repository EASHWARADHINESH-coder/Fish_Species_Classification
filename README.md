# 🐟 Fish Species Classification using Deep Learning

A Deep Learning-based web application that classifies fish species from images using a **MobileNet Convolutional Neural Network (CNN)** and an interactive **Streamlit interface**.

This project demonstrates the complete pipeline of **computer vision → model training → deployment**.

---

## 📌 Project Objective

The objective of this project is to build an intelligent system capable of:

- Identifying fish species from images  
- Using transfer learning with a pretrained deep learning model  
- Providing real-time predictions through a web application  

### 🌊 Real-world Applications

- Marine research  
- Fishery management  
- Biodiversity monitoring  
- Educational tools  

---

## 🧠 Deep Learning Model

| Feature | Description |
|---------|-------------|
| Model Type | Convolutional Neural Network (CNN) |
| Base Model | MobileNet (Pretrained on ImageNet) |
| Technique | Transfer Learning |
| Input Size | 224 × 224 RGB images |
| Output | Multi-class fish species classification |
| Framework | TensorFlow / Keras |

### 🔹 Why MobileNet?

- Lightweight and efficient  
- Optimized for image recognition  
- High accuracy with low computational cost  
- Ideal for web deployment  

---

## 🚀 Web Application (Streamlit)

The trained model is deployed using **Streamlit**, providing an easy-to-use interface.

### ✨ Application Features

- Upload fish image  
- Automatic preprocessing  
- Species prediction  
- Confidence score display  
- Probability scores for all classes  
- Fast and interactive UI  

---

## 🔄 System Workflow

1. User uploads fish image  
2. Image resized to 224×224  
3. MobileNet preprocessing applied  
4. Model predicts probabilities  
5. Highest probability → predicted species  
6. Confidence score displayed  

---

## 🛠️ Technologies Used

- Python  
- TensorFlow  
- Keras  
- Streamlit  
- NumPy  
- Pillow  

---

## 📊 Machine Learning Concepts

- Convolutional Neural Networks  
- Transfer Learning  
- Image Preprocessing  
- Softmax Classification  
- Model Deployment  

---

## 🎯 Future Improvements

- Add Grad-CAM visualization  
- Improve dataset size and balance  
- Add species information panel  
- Cloud-based model storage  
- Mobile-friendly interface  

---

## 👨‍💻 Author

**Eashwaradhinesh K**  
Deep Learning & Data Science Enthusiast  

---

## 📜 License

This project is licensed under the **MIT License**.

---
