import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import random

# Load model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Disease classes
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

# Remedies dictionary (keys simplified)
remedies = {
    "tomato_early_blight": {
        "natural": {
            "info": "Remove affected leaves and spray neem oil regularly. Maintain good air circulation between plants.",
            "products": [
                "Neem Oil", "Compost Tea Spray", "Baking Soda Solution",
                "Garlic Spray", "Aloe Vera Extract", "Cinnamon Oil Spray"
            ]
        },
        "pesticide": {
            "info": "Apply copper-based fungicides or chlorothalonil spray as per instructions.",
            "products": [
                "Copper Oxychloride", "Chlorothalonil Fungicide", "Mancozeb",
                "Carbendazim", "Copper Hydroxide", "Azoxystrobin"
            ]
        }
    },
    "potato_late_blight": {
        "natural": {
            "info": "Ensure proper spacing and sunlight exposure. Use garlic extract spray on affected areas.",
            "products": [
                "Garlic Extract Spray", "Neem Oil", "Turmeric Spray",
                "Trichoderma Bio-fungicide", "Aloe Vera Extract"
            ]
        },
        "pesticide": {
            "info": "Spray with metalaxyl or cymoxanil-based fungicides to control the spread.",
            "products": [
                "Metalaxyl 8% + Mancozeb 64%", "Cymoxanil 8% + Mancozeb 64%", 
                "Dimethomorph", "Propamocarb", "Zineb"
            ]
        }
    },
    "pepper_bell_bacterial_spot": {
        "natural": {
            "info": "Use copper-based organic sprays and avoid overhead watering.",
            "products": [
                "Copper Soap Fungicide", "Neem Oil", "Compost Tea Spray",
                "Garlic Extract", "Trichoderma Bio-fungicide"
            ]
        },
        "pesticide": {
            "info": "Spray with copper hydroxide or streptomycin formulations.",
            "products": [
                "Copper Hydroxide Spray", "Streptomycin Sulfate",
                "Copper Oxychloride", "Kasugamycin", "Oxytetracycline"
            ]
        }
    },
    "tomato_spider_mites_two_spotted_spider_mite": {
        "natural": {
            "info": "Spray with neem oil or insecticidal soap. Maintain humidity around plants.",
            "products": [
                "Neem Oil", "Insecticidal Soap", "Clove Oil Spray",
                "Rosemary Oil Spray", "Horticultural Oil"
            ]
        },
        "pesticide": {
            "info": "Apply abamectin or spiromesifen-based miticides.",
            "products": [
                "Abamectin 1.9% EC", "Spiromesifen 22.9% SC", "Fenpyroximate",
                "Bifenazate", "Hexythiazox"
            ]
        }
    }
}

# Helper function to normalize class names
def normalize_name(name):
    return name.lower().replace("__", "_").replace("___", "_").replace(" ", "_")

# Streamlit UI
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

    # Safe check for index error
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

        if normalized_class in remedies:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🌿 Natural Remedy"):
                    st.subheader("Natural Remedy")
                    st.write(remedies[normalized_class]["natural"]["info"])
                    st.markdown("**Recommended Natural Products:**")
                    random_natural = random.sample(remedies[normalized_class]["natural"]["products"], k=3)
                    for p in random_natural:
                        st.markdown(f"- 🔹 [{p} on Amazon](https://www.amazon.in/s?k={p.replace(' ', '+')})")

            with col2:
                if st.button("🧴 Pesticide"):
                    st.subheader("Pesticide Treatment")
                    st.write(remedies[normalized_class]["pesticide"]["info"])
                    st.markdown("**Suggested Pesticide Products:**")
                    random_pesticides = random.sample(remedies[normalized_class]["pesticide"]["products"], k=3)
                    for p in random_pesticides:
                        st.markdown(f"- 💊 [{p} on Amazon](https://www.amazon.in/s?k={p.replace(' ', '+')})")
