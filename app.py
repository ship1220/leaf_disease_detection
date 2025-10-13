import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import time

# --- Load model ---
model = tf.keras.models.load_model("plant_disease_model.keras")

# --- Class names ---
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

# --- Remedies dictionary ---
disease_remedies = {
    "Pepper__bell___Bacterial_spot": {
        "pesticide": "Copper-based bactericides (e.g. Copper oxychloride) applied every 7–10 days.",
        "natural": "Use neem oil spray, ensure leaves stay dry (avoid overhead watering), remove infected leaves."
    },
    "Potato___Early_blight": {
        "pesticide": "Use fungicides like Mancozeb or Chlorothalonil.",
        "natural": "Spray with neem oil or compost tea weekly; ensure good aeration."
    },
    "Potato___Late_blight": {
        "pesticide": "Use Metalaxyl, or copper fungicides.",
        "natural": "Use garlic extract sprays, remove infected foliage, avoid waterlogging."
    },
    "Tomato_Target_Spot": {
        "pesticide": "Use fungicides such as Chlorothalonil or Mancozeb.",
        "natural": "Remove lower leaves, spray neem oil, avoid wet leaf surfaces."
    },
    "Tomato_Tomato_mosaic_virus": {
        "pesticide": "No effective chemical treatment for virus — manage vectors (insecticides against thrips/aphids).",
        "natural": "Remove infected plants, disinfect tools, introduce predators, control insect vectors."
    },
    "Tomato_Tomato_YellowLeaf__Curl_Virus": {
        "pesticide": "Use insecticides (e.g. Imidacloprid) to control whiteflies.",
        "natural": "Introduce natural predators (ladybugs), use yellow sticky traps, remove weeds."
    },
    "Tomato_Bacterial_spot": {
        "pesticide": "Copper-based bactericides every 7 days.",
        "natural": "Spray baking soda + neem oil, avoid overhead watering, crop rotation."
    },
    "Tomato_Early_blight": {
        "pesticide": "Use fungicides like Mancozeb or Chlorothalonil.",
        "natural": "Spray neem oil or compost tea every 5–7 days, remove infected parts."
    },
    "Tomato_Late_blight": {
        "pesticide": "Use copper fungicides or systemic fungicides.",
        "natural": "Remove infected leaves, spray garlic or neem mixture."
    },
    "Tomato_Leaf_Mold": {
        "pesticide": "Use fungicides such as Chlorothalonil or Mancozeb.",
        "natural": "Ensure good ventilation, avoid moisture accumulation, spray neem oil."
    },
    "Tomato_Septoria_leaf_spot": {
        "pesticide": "Use fungicides like Mancozeb or Chlorothalonil.",
        "natural": "Spray neem oil, remove crop debris, rotate crops."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "pesticide": "Use miticides like Abamectin or Dicofol.",
        "natural": "Spray neem oil, insecticidal soap, increase humidity, release predatory mites."
    }
}

# --- Product suggestions (neem / botanical / pesticide) ---
# We'll use product_query to fetch some example products
from streamlit import session_state  # optional if needed

# (You’ll need product_query at top-level, but I'll show inside logic here.)
# See below for how to fetch product suggestions with web tool.

def show_product_suggestions():
    st.subheader("🛒 Suggested Products (You may purchase)")
    st.write("These are example botanical / pesticide / neem-oil products you can look at:")
    # Example static links (you could fetch dynamically)
    st.markdown("- Green Dews Neem Oil (200 ml) — Flipkart listing :contentReference[oaicite:0]{index=0}")
    st.markdown("- Pestly Pure Neem Oil Mix (250 ml) — Flipkart :contentReference[oaicite:1]{index=1}")
    st.markdown("- RAVK KVAR Organic Neem Oil (5 L) — Flipkart :contentReference[oaicite:2]{index=2}")
    st.markdown("- Leafy Tales Neem Oil 600 ml — Flipkart :contentReference[oaicite:3]{index=3}")
    st.markdown("- UGAOO Neem Oil Spray — Flipkart :contentReference[oaicite:4]{index=4}")
    st.markdown("- MyOwnGarden Organic Neem Oil 1 L — Flipkart :contentReference[oaicite:5]{index=5}")
    st.markdown("- A R S H Neem Oil 500 ml — Flipkart :contentReference[oaicite:6]{index=6}")
    st.markdown("*(Note: Always verify product labels, regulations, and safety before use.)*")

# --- Page setup ---
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

st.title("🌿 Plant Disease Detection & Remedies")
st.write("Upload a leaf image. If disease is detected, click a button to see **Natural Remedy** or **Pesticide Suggestion**.")

uploaded_file = st.file_uploader("Upload leaf image (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(image, caption="Uploaded Leaf", use_column_width=True)
    with col2:
        with st.spinner("Analyzing image... 🔎"):
            time.sleep(1)
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = predictions[0][predicted_index]

        st.subheader("📋 Prediction Result")
        if "healthy" in predicted_class.lower():
            st.success(f"✅ The plant is **Healthy** (Confidence: {confidence:.2f})")
        else:
            st.error(f"❌ Diseased: **{predicted_class.replace('_', ' ')}** (Confidence: {confidence:.2f})")

            # Show buttons
            col_button1, col_button2 = st.columns(2)
            with col_button1:
                if st.button("Natural Remedy"):
                    remedy = disease_remedies.get(predicted_class, {}).get("natural")
                    if remedy:
                        st.markdown(f"**🌱 Natural Remedy**:\n{remedy}")
                    else:
                        st.info("No natural remedy info available.")

            with col_button2:
                if st.button("Pesticide Suggestion"):
                    remedy = disease_remedies.get(predicted_class, {}).get("pesticide")
                    if remedy:
                        st.markdown(f"**💊 Pesticide Suggestion**:\n{remedy}")
                    else:
                        st.info("No pesticide suggestion available.")

            # After showing remedy, show product suggestions
            show_product_suggestions()

# Sidebar or footer
st.sidebar.title("ℹ️ Info & Disclaimer")
st.sidebar.markdown(
    """
    - The remedy suggestions are **informational only**.  
    - Always check **local agricultural guidelines**, **product labels**, and **safety instructions**.  
    - Use protective gear during spraying and follow dosage instructions.  
    - This app does *not* substitute expert advice.
    """
)
