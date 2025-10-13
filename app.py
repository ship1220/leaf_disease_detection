import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import time

# Load model
model = tf.keras.models.load_model("plant_disease_model1.keras")

# Class names
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato_Target_Spot',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]

# Page setup
st.set_page_config(page_title="Plant Disease Detector 🌱", page_icon="🌿", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #F8FFF6;
        padding: 1.5rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("🌿 Plant Disease Detection System")
st.write("Upload a clear image of a plant leaf to detect if it's **healthy** or affected by a **disease**.")

# File uploader
uploaded_file = st.file_uploader("📸 Upload your image (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

    with col2:
        with st.spinner("Analyzing the image... 🔍"):
            time.sleep(1)  # for visual feedback
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = predictions[0][predicted_index]

        st.subheader("🌾 Prediction Results")

        if "healthy" in predicted_class.lower():
            st.success(f"✅ The plant is **Healthy!** (Confidence: {confidence:.2f})")
        else:
            st.error(f"❌ The plant is **Diseased!**")
            st.warning(f"**Detected Disease:** {predicted_class.replace('_', ' ')}")
            st.progress(float(confidence))

        # Optional: Show top 3 predictions
        top3_idx = predictions[0].argsort()[-3:][::-1]
        st.write("### 🔝 Top 3 Predictions")
        for i in top3_idx:
            st.write(f"- {class_names[i]}: **{predictions[0][i]:.2f}**")

        st.info("💡 Tip: Ensure good lighting and clear focus for best results.")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    """
    **Model:** CNN-based Plant Disease Detector  
    **Framework:** TensorFlow + Streamlit  
    **Author:** Shipra 🌸  
    ---
    Upload images of **Pepper**, **Potato**, or **Tomato** leaves  
    to detect diseases instantly!
    """
)
