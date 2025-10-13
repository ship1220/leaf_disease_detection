import gdown
import os
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import random

# -------------------------
# Download Model from Google Drive
# -------------------------
output_path = "plant_disease_model.keras"
file_id = "19E5GXc8fUO-afXLLOLgtCYnox408k3_O"

if not os.path.exists(output_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, output_path, quiet=False)

model = tf.keras.models.load_model(output_path)

# -------------------------
# Disease classes
# -------------------------
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

# -------------------------
# Remedies dictionary (add default for unmatched diseases)
# -------------------------
remedies = {
    "tomato_early_blight": {
        "natural": {
            "info": "Remove affected leaves and spray neem oil regularly. Maintain good air circulation.",
            "products": ["Neem Oil", "Compost Tea Spray", "Baking Soda Solution"]
        },
        "pesticide": {
            "info": "Apply copper-based fungicides or chlorothalonil spray as per instructions.",
            "products": ["Copper Oxychloride", "Chlorothalonil Fungicide", "Mancozeb"]
        }
    },
    "potato_late_blight": {
        "natural": {
            "info": "Ensure proper spacing and sunlight exposure. Use garlic extract spray.",
            "products": ["Garlic Extract Spray", "Neem Oil", "Turmeric Spray"]
        },
        "pesticide": {
            "info": "Spray with metalaxyl or cymoxanil-based fungicides to control the spread.",
            "products": ["Metalaxyl + Mancozeb", "Cymoxanil + Mancozeb", "Dimethomorph"]
        }
    },
    "pepper_bell_bacterial_spot": {
        "natural": {
            "info": "Use copper-based organic sprays and avoid overhead watering.",
            "products": ["Copper Soap Fungicide", "Neem Oil", "Compost Tea Spray"]
        },
        "pesticide": {
            "info": "Spray with copper hydroxide or streptomycin formulations.",
            "products": ["Copper Hydroxide Spray", "Streptomycin Sulfate", "Copper Oxychloride"]
        }
    },
    "tomato_spider_mites_two_spotted_spider_mite": {
        "natural": {
            "info": "Spray with neem oil or insecticidal soap. Maintain humidity around plants.",
            "products": ["Neem Oil", "Insecticidal Soap", "Clove Oil Spray"]
        },
        "pesticide": {
            "info": "Apply abamectin or spiromesifen-based miticides.",
            "products": ["Abamectin 1.9% EC", "Spiromesifen 22.9% SC", "Fenpyroximate"]
        }
    },
    # Default remedies for unmatched diseases
    "default": {
        "natural": {
            "info": "Maintain plant hygiene, remove infected leaves, and use organic sprays like neem oil.",
            "products": ["Neem Oil", "Garlic Spray", "Aloe Vera Extract"]
        },
        "pesticide": {
            "info": "Use recommended pesticides as per local agricultural guidelines.",
            "products": ["General Fungicide", "Copper Oxychloride", "Mancozeb"]
        }
    }
}

# -------------------------
# Helper function
# -------------------------
def normalize_name(name):
    return name.lower().replace("__", "_").replace("___", "_").replace(" ", "_")

# -------------------------
# Streamlit UI
# -------------------------
st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image to check if it is healthy or diseased.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])

    # Safe index check
    if predicted_index >= len(class_names):
        st.error(f"⚠️ Predicted index {predicted_index} out of range.")
        st.stop()

    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]
    normalized_class = normalize_name(predicted_class)

    if "healthy" in predicted_class.lower():
        st.success(f"The plant is healthy! 🌿 (Confidence: {confidence:.2f})")
    else:
        st.error(f"The plant is diseased! ❌\n**Disease:** {predicted_class}\n**Confidence:** {confidence:.2f}")

        # Fetch remedies or use default
        disease_data = remedies.get(normalized_class, remedies["default"])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌿 Natural Remedy"):
                st.subheader("Natural Remedy")
                st.write(disease_data["natural"]["info"])
                st.markdown("**Recommended Natural Products:**")
                random_natural = random.sample(disease_data["natural"]["products"], 
                                               k=min(3, len(disease_data["natural"]["products"])))
                for p in random_natural:
                    st.markdown(f"- 🔹 [{p} on Amazon](https://www.amazon.in/s?k={p.replace(' ', '+')})")

        with col2:
            if st.button("🧴 Pesticide"):
                st.subheader("Pesticide Treatment")
                st.write(disease_data["pesticide"]["info"])
                st.markdown("**Suggested Pesticide Products:**")
                random_pesticides = random.sample(disease_data["pesticide"]["products"], 
                                                 k=min(3, len(disease_data["pesticide"]["products"])))
                for p in random_pesticides:
                    st.markdown(f"- 💊 [{p} on Amazon](https://www.amazon.in/s?k={p.replace(' ', '+')})")
