"""
AI Plant Disease Diagnosis — Streamlit Dashboard
==================================================
Deployment-ready inference app for the MobileNetV2 leaf-disease classifier,
styled to match the PlantAI Diagnosis dashboard design.

Requirements:
    - google_finetuned_model.keras   (or change MODEL_PATH below)
    - class_indices.json             (generated during training)
    - disease_info.json              (disease details schema)

Run:
    streamlit run app.py
"""

import os
import json
from urllib.parse import quote_plus

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "google_finetuned_model.keras"
FALLBACK_MODEL_PATH = "best_finetuned_model.keras"
CLASS_INDICES_PATH = "class_indices.json"
IMAGE_SIZE = (224, 224)
TOP_K = 3

st.set_page_config(
    page_title="PlantAI Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource(show_spinner="Loading AI Model...")
def load_trained_model():
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
    if not os.path.exists(path):
        st.error(f"Model file not found. Expected '{MODEL_PATH}' or '{FALLBACK_MODEL_PATH}' in the app directory.")
        st.stop()
    return tf.keras.models.load_model(path)


@st.cache_data(show_spinner=False)
def load_class_names():
    if not os.path.exists(CLASS_INDICES_PATH):
        st.error(f"'{CLASS_INDICES_PATH}' not found in the app directory.")
        st.stop()
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    ordered = sorted(class_indices.items(), key=lambda kv: kv[1])
    return [name for name, _ in ordered]


@st.cache_data(show_spinner=False)
def load_disease_info():
    if not os.path.exists("disease_info.json"):
        st.error("'disease_info.json' not found in the app directory.")
        st.stop()
    with open("disease_info.json", "r", encoding="utf-8") as f:
        return json.load(f)


model = load_trained_model()
class_names = load_class_names()
disease_info = load_disease_info()

# ============================================================
# HELPERS
# ============================================================

def amazon_link(product_name: str) -> str:
    return f"https://www.amazon.in/s?k={quote_plus(product_name)}"


def predict(pil_image: Image.Image):
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:TOP_K]
    return [(class_names[i], float(preds[i])) for i in top_idx]


def render_products(products):
    if not products:
        st.caption("No targeted products listed for this condition.")
        return
    for name in products:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{name}**")
        with c2:
            st.link_button("Amazon", amazon_link(name), use_container_width=True)

# ============================================================
# SIDEBAR CONFIGURATION (SIMPLE TEXT FORMAT)
# ============================================================

with st.sidebar:
    st.title("PlantAI")
    st.caption("v1.0.0 · MobileNetV2 Fine-tuned")
    st.divider()
    
    st.markdown("### AI Model Architecture")
    
    AI_STATS = [
        ("Base Architecture", "MobileNetV2 (ImageNet Weights)"),
        ("Training Routine", "Transfer Learning + Soft Fine-Tuning"),
        ("Total Classes Matrix", f"{len(class_names)} Registered Disease Topologies"),
        ("Supported Crops", "3 Key Substrates (Tomato, Potato, Pepper)"),
        ("Fixed Input Aspect", "224 x 224 px Tensor Matrix"),
        ("Preprocessing Pipeline", "MobileNetV2 standard scaling"),
    ]
    
    # Modern text arrangement instead of stacked boxes
    for label, value in AI_STATS:
        st.markdown(f"**{label}**  \n{value}")
        st.write("")
        
    st.divider()
    st.markdown("### Dataset Summary")
    st.caption("Dataset Platform: PlantVillage Core Infrastructure + Supplementary Synthetic Web Field Captures.")

# ============================================================
# APP HEADER & HERO
# ============================================================

st.subheader("PlantAI Diagnosis ", divider="green")

with st.container(border=True):
    st.markdown("### AI-Powered · Real-time Plant Diagnostics")
    st.write(
        "Upload a tomato, potato, or bell pepper leaf image and receive an instant "
        "AI-powered health diagnosis alongside recommended remediation strategies."
    )
    st.pills("Supported Crops Focus", ["Tomato", "Potato", "Bell Pepper"], disabled=True)

# ============================================================
# UPLOAD + MAIN RESULT SECTION
# ============================================================

st.write("### Upload & Diagnose")
col_upload, col_result = st.columns(2)

uploaded_file = None
with col_upload:
    with st.container(border=True):
        st.markdown("**Upload Leaf Image**")
        uploaded_file = st.file_uploader(
            "Drop your leaf image here or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

results = None
info = None

with col_result:
    with st.container(border=True):
        st.markdown("**Diagnosis Result**")
        
        if uploaded_file is None:
            st.write("")
            st.info("Please upload a leaf image on the left panel to trigger the AI diagnostic inference workflow.")
            st.write("")
        else:
            with st.spinner("Running AI model inference structures..."):
                results = predict(image)

            top_class, top_conf = results[0]
            info = disease_info.get(top_class)
            healthy = info["healthy"] if info else False
            display_name = info["display_name"] if info else top_class
            
            # Metric Block
            st.metric(label="AI Diagnostic Confidence", value=f"{top_conf*100:.2f}%")
            st.progress(top_conf)
            
            # Diagnostic status
            if healthy:
                st.success(f"### Healthy Asset: {display_name}")
            else:
                st.error(f"### Action Required: {display_name}")
                
            if info:
                st.markdown(f"*{info['description']}*")

st.divider()

# ============================================================
# DETAILED METRIC BREAKDOWNS (POST INFERENCE ONLY)
# ============================================================

if results and info:
    healthy = info["healthy"]
    
    # Primary Diagnosis Display Banner
    with st.container(border=True):
        status_label = "ASSET SYSTEM HEALTHY" if healthy else "PATHOLOGY THREAT DETECTED"
        st.markdown(f"`{status_label}`")
        st.markdown(f"## Primary Prediction: **{info['display_name']}**")
        st.markdown(f"{info['description']}")
    
    # Top-3 Probabilities Layout
    st.write("### Top 3 Probabilities Distribution")
    for idx, (name, conf) in enumerate(results):
        entry = disease_info.get(name, {})
        label = entry.get("display_name", name)
        r_healthy = entry.get("healthy", False)
        health_tag = "Healthy" if r_healthy else "Diseased"
        
        c_idx, c_lbl, c_prg = st.columns([0.5, 4, 6])
        c_idx.markdown(f"`#{idx+1}`")
        c_lbl.markdown(f"**{label}**  \n*{health_tag}*")
        c_prg.write(f"Confidence: **{conf*100:.1f}%**")
        c_prg.progress(conf)
        st.write("")

    # Deep-Dive Disease / Actionable Insight Panels
    if not healthy:
        st.write("### Pathological Insight & Metadata Profile")
        
        # Collapsible Modern Accordions
        with st.expander("Primary Vectors & Visual Symptoms Profile", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Primary Vectors & Causes**")
                for item in info.get("causes", []):
                    st.markdown(f"- {item}")
            with c2:
                st.markdown("**Visual Symptoms Profile**")
                for item in info.get("symptoms", []):
                    st.markdown(f"- {item}")

        with st.expander("Environmental Factors & Preventative Cultivation", expanded=False):
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**High Risk Environmental Climate**")
                for item in info.get("climate", []):
                    st.markdown(f"- {item}")
            with c4:
                st.markdown("**Preventative Structural Cultivation**")
                for item in info.get("prevention", []):
                    st.markdown(f"- {item}")

        with st.expander("Impact Reduction Directives", expanded=False):
            for item in info.get("reduce_impact", []):
                st.markdown(f"- {item}")

        # Actionable Remedy Links Marketplace
        st.write("### Integrated Action & Remediation Treatments")
        rc1, rc2 = st.columns(2)
        with rc1:
            with st.container(border=True):
                st.markdown("#### Organic & Natural Remediation")
                render_products(info.get("natural_remedies", []))
        with rc2:
            with st.container(border=True):
                st.markdown("#### Chemical Treatments & Controls")
                render_products(info.get("chemical_remedies", []))
    else:
        # Maintenance profile for validated clean leaves
        st.write("### Crop System Maintenance Guidelines")
        with st.container(border=True):
            st.markdown("#### Clean Culture & Best Practices")
            for item in info.get("prevention", []):
                st.markdown(f"- {item}")
        
        with st.container(border=True):
            st.markdown("#### Ongoing Organic Preventative Maintenance")
            render_products(info.get("natural_remedies", []))

st.divider()

# ============================================================
# STANDARD APPLICATION FOOTER (BOXLESS & MINIMAL)
# ============================================================

st.markdown("**PlantAI Diagnosis Dashboard Environment · v1.0.0**")
st.caption(
    "Disclaimer Notice: All machine learning inference profiles, telemetry matrices, "
    "and recommendations generated by this app framework are formulated for data visualization purposes only. "
    "Always cross-reference target data structures with a certified master agriculturalist before deploying chemical treatments."
)