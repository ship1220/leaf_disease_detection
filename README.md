# 🌿 Leaf Disease Detection using MobileNetV2  

## 🔍 Overview  
This project applies **deep learning and computer vision** to detect plant leaf diseases from images.  
Using **MobileNetV2**, a lightweight and efficient CNN architecture, the model classifies plant leaves into multiple disease categories and provides confidence scores for each prediction.  

The trained model is deployed through a **Streamlit web app**, allowing users to upload a leaf image and get real-time predictions — supporting early detection and sustainable agriculture.  

---

## 🎯 Objective  
- Automate plant leaf disease detection using deep learning.  
- Achieve high classification accuracy with an efficient architecture.  
- Provide a simple, accessible web interface for real-time use.  

---

## 🧠 Model Architecture  

### Model Used: **MobileNetV2 (Transfer Learning)**  
- **Base Model:** MobileNetV2 pre-trained on ImageNet  
- **Input Size:** 224 × 224 × 3  
- **Unfrozen Layers:** Last 60 layers for fine-tuning  
- **Added Layers:**  
  - GlobalAveragePooling2D  
  - Dense(128, activation='relu')  
  - Dropout(0.5)  
  - Dense(num_classes, activation='softmax')  

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  
**Accuracy:** ~97% on PlantVillage dataset  

---

## 📊 Dataset  
- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Images:** 50,000+ labeled leaf images (healthy & diseased)  
- **Classes:** 14+ disease categories  
- **Preprocessing:**  
  - Image resizing to 224×224  
  - Normalization to [0,1]  
  - Data augmentation (rotation, zoom, flip)  

---

## ⚙️ Implementation Steps  
1. **Data Loading:** Mounted dataset from Google Drive in Colab.  
2. **Preprocessing:** Used `ImageDataGenerator` for normalization and augmentation.  
3. **Model Building:** Loaded and customized MobileNetV2 using TensorFlow/Keras.  
4. **Training:** Trained for 15 epochs using Colab GPU.  
5. **Evaluation:** Analyzed accuracy, precision, recall, and confusion matrix.  
6. **Deployment:** Integrated model into Streamlit for real-time prediction.  

---

## 💻 Streamlit App  
Upload a leaf image (`.jpg` / `.png`) to get:  
- Predicted disease name 🌿  
- Model confidence score 📈  
- Health status indicator (✅ Healthy / ⚠️ Diseased)  

**Run locally:**
```bash
streamlit run app.py
