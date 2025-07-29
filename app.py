import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# Set page config
st.set_page_config(page_title="Potato Leaf Disease Detection", layout="centered")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "HOME"

if "model" not in st.session_state:
    st.session_state.model = None

# Detect light or dark mode (Streamlit >=1.46)
theme = st.get_option("theme.base")
is_dark = theme == "dark"

# Download model if not exists
MODEL_PATH = "potato_model.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=1DBXxD2KYRWi7AR8-MklXet5d1B1jydqI"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

# Sidebar navigation
st.session_state.page = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION", "PROJECT DETAILS"])

# HOME PAGE
if st.session_state.page == "HOME":
    st.title("🍃 Potato Leaf Disease Detection")
    st.markdown("""
    Welcome to the Potato Disease Detection App.

    Navigate to **DISEASE RECOGNITION** to identify potato leaf diseases using AI.
    
    Developed using TensorFlow and Streamlit.
    """)

# DISEASE RECOGNITION
elif st.session_state.page == "DISEASE RECOGNITION":
    st.title("🔍 Disease Recognition")

    uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            if st.session_state.model is None:
                st.session_state.model = load_model()

            model = st.session_state.model
            img = image.resize((256, 256))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)[0]
            classes = ["Early Blight", "Healthy", "Late Blight"]
            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.success(f"**Prediction:** {predicted_class} ({confidence * 100:.2f}% confidence)")

# PROJECT DETAILS
elif st.session_state.page == "PROJECT DETAILS":
    st.title("📁 Project Details")
    st.markdown("""
    ### About the Project
    This app uses a Convolutional Neural Network (CNN) trained on a dataset of potato leaf images to classify:

    - Early Blight
    - Late Blight
    - Healthy

    ### Technologies Used
    - TensorFlow / Keras
    - Streamlit
    - Google Drive (model hosting)

    ### Team
    Developed by [Your Team Name or Institution].
    """)
