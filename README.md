## Demo: https://leafdiseasedetection-bnscpebysmu5rfffmwalwf.streamlit.app/

# AI-Powered Plant Leaf Disease Detection & Diagnosis

An AI-powered web application that detects diseases in **Tomato, Potato, and Bell Pepper** leaves using **MobileNetV2** with Transfer Learning. The application provides disease diagnosis, confidence score, causes, symptoms, prevention methods, and treatment recommendations through an interactive Streamlit interface.

## Features

* Upload a leaf image for instant diagnosis
* AI-based disease classification (15 classes)
* Confidence score with Top-3 predictions
* Disease description, causes, and symptoms
* Suitable climate and prevention tips
* Natural and chemical treatment recommendations
* Suggested agricultural products

## Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)
* Streamlit
* OpenCV & Pillow
* NumPy

## Dataset

The model was trained using a combination of:

* PlantVillage Dataset
* Real-world Field Images
* Curated Google Images (for domain adaptation)

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Supported Diseases

* Bell Pepper (Healthy, Bacterial Spot)
* Potato (Healthy, Early Blight, Late Blight)
* Tomato (Healthy, Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Target Spot, Spider Mites, Yellow Leaf Curl Virus, Mosaic Virus)

