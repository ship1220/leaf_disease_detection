🌿 Leaf Disease Detection using MobileNetV2
🔍 Overview

This project uses deep learning and computer vision to automatically detect plant leaf diseases from images.
Built using MobileNetV2, a lightweight yet powerful CNN architecture, the model classifies plant leaves into multiple disease categories and provides confidence scores for each prediction.

The system is deployed as a Streamlit web application, allowing users to upload leaf images and receive real-time classification results — helping farmers and researchers quickly identify crop diseases.

🎯 Project Objective

Automate the process of plant disease detection using image recognition.

Achieve high accuracy while keeping the model lightweight for real-world deployment.

Provide an easy-to-use web interface for real-time predictions.

🧠 Model & Architecture
Model Used: MobileNetV2 (Transfer Learning)

Base Model: MobileNetV2 (pre-trained on ImageNet)

Input Size: 224 × 224 × 3

Fine-tuned Layers: Last 60 layers unfrozen for task-specific learning

Added Layers:

GlobalAveragePooling2D

Dense(128, activation='relu')

Dropout(0.5)

Dense(num_classes, activation='softmax')

Optimizer: Adam
Loss Function: Categorical Crossentropy
Accuracy Achieved: ~97% on PlantVillage dataset

📊 Dataset

Source: PlantVillage Dataset

Size: 50,000+ labeled images of healthy and diseased plant leaves

Categories: 14+ disease types across different plant species

Preprocessing:

Image resizing (224×224)

Normalization (scaling pixel values to [0,1])

Data augmentation (rotation, zoom, flipping)

⚙️ Implementation Steps

Data Loading – Mounted Google Drive and accessed dataset in Colab.

Preprocessing – Resized and normalized images using ImageDataGenerator.

Model Building – Loaded MobileNetV2, added dense layers, and compiled the model.

Training – Conducted on Google Colab using GPU (15 epochs).

Evaluation – Measured performance using accuracy, precision, recall, and confusion matrix.

Deployment – Integrated trained model into a Streamlit web app for real-time prediction.

💻 Streamlit App

Upload a leaf image (.jpg/.png).

The app displays:

Predicted disease name 🌿

Model confidence score 📈

Visual feedback (Healthy ✅ / Diseased ⚠️).

Run locally:

streamlit run app.py

🧰 Technologies Used

Programming Language: Python

Frameworks & Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Streamlit

Environment: Google Colab (GPU-accelerated)

Dataset: PlantVillage

📁 Repository Structure
Leaf-Disease-Detection/
│
├── leaf_disease_model.keras       # Trained model file
├── app.py                         # Streamlit web app
├── dataset/                       # Dataset (PlantVillage or subset)
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
└── utils/                         # Helper scripts (preprocessing, visualization)

📈 Results
Metric	Training	Validation
Accuracy	98.2%	97.0%
Loss	0.07	0.11

Confusion matrix and class-wise accuracy plots confirmed consistent model performance across all disease categories.

🚀 Future Enhancements

Extend support for additional plant species.

Integrate real-time detection via smartphone camera.

Build an offline mobile version using TensorFlow Lite.

Add disease-specific remedy recommendations.
