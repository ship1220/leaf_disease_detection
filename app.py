import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import time

# --- Load Model ---
model = tf.keras.models.load_model("plant_disease_model.keras")

# --- Class Names ---
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

# --- Remedies Dictionary ---
disease_remedies = {
    "Pepper__bell___Bacterial_spot": {
        "pesticide": "Copper-based bactericides such as Copper Oxychloride.",
        "natural": "Use neem oil spray and avoid overhead watering."
    },
    "Potato___Early_blight": {
        "pesticide": "Mancozeb 75% WP or Chlorothalonil fungicide.",
        "natural": "Apply compost tea or neem oil once a week."
    },
    "Potato___Late_blight": {
        "pesticide": "Metalaxyl or Copper oxychloride fungicide.",
        "natural": "Spray garlic extract and avoid excess humidity."
    },
    "Tomato_Target_Spot": {
        "pesticide": "Chlorothalonil or mancozeb fungicide.",
        "natural": "Prune lower leaves and use neem oil spray weekly."
    },
    "Tomato_Tomato_mosaic_virus": {
        "pesticide": "No chemical control available.",
        "natural": "Remove infected plants and disinfect tools regularly."
    },
    "Tomato_Tomato_YellowLeaf__Curl_Virus": {
        "pesticide": "Use insecticides like Imidacloprid to control whiteflies.",
        "natural": "Introduce natural predators (ladybugs) and remove weeds."
    },
    "Tomato_Bacterial_spot": {
        "pesticide": "Copper-based bactericides every 7 days.",
        "natural": "Apply baking soda and neem oil solution."
    },
    "Tomato_Early_blight": {
        "pesticide": "Mancozeb 75% WP or Chlorothalonil fungicide.",
        "natural": "Use compost tea or neem oil every 5–7 days."
    },
    "Tomato_Late_blight": {
        "pesticide": "Copper oxychloride or Metalaxyl.",
        "natural": "Remove infected leaves and use garlic spray weekly."
    },
    "Tomato_Leaf_Mold": {
        "pesticide": "Chlorothalonil or mancozeb fungicide.",
        "natural": "Improve ventilation and reduce leaf wetness."
    },
    "Tomato_Septoria_leaf_spot": {
        "pesticide": "Mancozeb or Chlorothalonil every 10 days.",
        "natural": "Use neem oil spray and crop rotation."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "pesticide": "Abamectin or Dicofol miticide.",
        "natural": "Use neem oil or insecticidal soap solution."
    }
}

# --- Page Setup ---
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

# --- Title & Description ---
st.title("🌿 Plant Disease Detection System")
st.write("Upload a clear image of a plant leaf to detect if it's **healthy** or affected by a **disease**, and view possible remedies!")

# --- File Upload ---
uploaded_file = st.file_uploader("📸 Upload your leaf image (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

    with col2:
        with st.spinner("Analyzing the image... 🔍"):
            time.sleep(1)  # Just for visual feedback
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

            # --- Remedy Suggestions ---
            remedy = disease_remedies.get(predicted_class, None)
            if remedy:
                st.subheader("🧴 Recommended Treatments")
                st.write(f"**Pesticide Suggestion:** {remedy['pesticide']}")
                st.write(f"**Natural Remedy:** {remedy['natural']}")
            else:
                st.info("No remedy information available for this disease yet.")

        # --- Top 3 Predictions ---
        top3_idx = predictions[0].argsort()[-3:][::-1]
        st.write("### 🔝 Top 3 Predictions")
        for i in top3_idx:
            st.write(f"- {class_names[i]}: **{predictions[0][i]:.2f}**")

        st.info("💡 Tip: Ensure good lighting and clear focus for best results.")

# --- Sidebar Info ---
st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    """
    **Model:** CNN-based Plant Disease Detector  
    **Framework:** TensorFlow + Streamlit  
    **Created by:** Shipra 🌸  
    ---
    Upload images of **Pepper**, **Potato**, or **Tomato** leaves  
    to detect diseases and view remedies instantly!
    """
)
