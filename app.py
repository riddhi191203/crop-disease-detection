import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

st.set_page_config(
    page_title="Native Crop Disease Detection",
    page_icon="🌾",
    layout="wide"
)
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


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

st.markdown("<h2 style='text-align:center;color:#2E8B57;margin-bottom:0;'>🌾 Native Crop Disease Detection using CNN</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;margin-top:2px;'>Upload a leaf image to identify crop and disease.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📷 Leaf Image")
        st.image(image, width=250)


    with col2:
        st.markdown("#### 🌿 Prediction Panel")

        if st.button("🔍 Predict", use_container_width=True):
            with st.spinner("Analyzing..."):
                prediction, confidence = predict_disease(image)

            parts = prediction.split("_")
            crop = parts[0].title()
            disease = " ".join(parts[1:]).title()

            st.success(f"**Crop:** {crop}")

            if disease.lower() == "healthy":
                st.success("**Status:** Healthy Plant ✅")
            else:
                st.error(f"**Disease:** {disease}")

            st.progress(int(confidence))
            st.info(f"Confidence: **{confidence:.2f}%**")

st.markdown("<p style='text-align:center;font-size:12px;'>Deep Learning Powered Crop Health System</p>", unsafe_allow_html=True)
