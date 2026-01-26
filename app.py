import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌾",
    layout="centered"
)

MODEL_PATH = r"D:\ml_projects\crop_disease_detection_model_3.h5"
model = load_model(MODEL_PATH)

CLASS_NAMES = [
    'corn_blight',
    'corn_common_rust',
    'corn_gray_leaf_spot',
    'corn_healthy',
    'gram_anthracnose',
    'gram_healthy',
    'gram_leaf_crinckle',
    'gram_powdery_mildew',
    'gram_yellow_mosaic',
    'millet_blast',
    'millet_healthy',
    'millet_rust',
    'wheat_crown_and_root_rot',
    'wheat_healthy',
    'wheat_leaf_rust',
    'wheat_loose_smut'
]

def predict_disease(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100
    return predicted_class, confidence

st.sidebar.title("🌿 About")
st.sidebar.info(
    """
    AI model for detecting diseases in:
    - 🌽 Corn  
    - 🌱 Gram  
    - 🌾 Millet  
    - 🌾 Wheat  
    """
)

st.markdown("<h1 style='text-align:center;color:#2E8B57;'>🌾 Crop Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a leaf image to identify crop and disease.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
    st.markdown("---")

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            prediction, confidence = predict_disease(image)

        parts = prediction.split("_")
        crop = parts[0].title()
        disease = " ".join(parts[1:]).title()

        st.markdown("### 🌿 Prediction Result")

        if disease.lower() == "healthy":
            st.success(f"**Crop:** {crop}")
            st.success("**Status:** Healthy Plant ✅")
        else:
            st.success(f"**Crop:** {crop}")
            st.error(f"**Disease:** {disease}")

        st.progress(int(confidence))
        st.info(f"Confidence: **{confidence:.2f}%**")

st.markdown("<hr><p style='text-align:center;'>Deep Learning Powered Crop Health System</p>", unsafe_allow_html=True)
