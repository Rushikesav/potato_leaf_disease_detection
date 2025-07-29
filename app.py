import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --------------------
# Session State Setup
# --------------------
if "page" not in st.session_state:
    st.session_state.page = "HOME"

if "model" not in st.session_state:
    st.session_state.model = None

# --------------------
# Theme Detection (optional)
# --------------------
try:
    from streamlit import runtime
    ctx = runtime.scriptrunner.get_script_run_ctx()
    if ctx:
        theme = st.get_option("theme")
        is_dark = theme and theme.get("base") == "dark"
    else:
        is_dark = False
except:
    is_dark = False

# --------------------
# Page Navigation
# --------------------
st.sidebar.title("Navigation")
st.session_state.page = st.sidebar.selectbox("Go to", ["HOME", "DISEASE RECOGNITION"])

# --------------------
# Model Download
# --------------------
MODEL_URL = 'https://drive.google.com/uc?id=1-sUInltdD8Uocf3L_z5a8ZyoYJfr3Q0D'
MODEL_PATH = 'potato_model.h5'

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# --------------------
# HOME PAGE
# --------------------
if st.session_state.page == "HOME":
    st.title("🥔 Potato Leaf Disease Detection")
    st.markdown("""
        Welcome to the Potato Leaf Disease Detection App.

        Upload an image of a potato leaf to detect possible diseases using a trained deep learning model.

        Navigate to **DISEASE RECOGNITION** from the sidebar to get started.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/2f/Potato_Leaf_Curl.jpg", width=400)

# --------------------
# DISEASE RECOGNITION PAGE
# --------------------
elif st.session_state.page == "DISEASE RECOGNITION":
    st.title("🔍 Detect Disease from Leaf")

    uploaded_file = st.file_uploader("Upload an image of a potato leaf", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load model if not loaded
        if st.session_state.model is None:
            download_model()
            st.session_state.model = tf.keras.models.load_model(MODEL_PATH)

        # Preprocess image
        img_resized = image.resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = st.session_state.model.predict(img_array)
        class_names = ['Early Blight', 'Late Blight', 'Healthy']
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

    else:
        st.warning("Please upload an image to proceed.")
