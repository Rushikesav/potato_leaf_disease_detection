import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --------------------
# Initialization
# --------------------
if "page" not in st.session_state:
    st.session_state.page = "HOME"

if "model" not in st.session_state:
    model_path = "potato_disease_model.h5"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            gdown.download(
                "https://drive.google.com/uc?id=1g8U5cKqR1ImFDqDQEp4e3UADhIWTesv3",
                model_path,
                quiet=False
            )
    st.session_state.model = tf.keras.models.load_model(model_path)

# --------------------
# Detect Theme
# --------------------
try:
    user_theme = st.runtime.scriptrunner.get_script_run_ctx().session.client.theme
    is_dark = user_theme and user_theme.base == "dark"
except Exception:
    is_dark = False

bg_color = "#0E1117" if is_dark else "#FFFFFF"
text_color = "#FAFAFA" if is_dark else "#000000"

# --------------------
# Page Selection
# --------------------
st.sidebar.title("Potato Leaf Disease Detection")
st.session_state.page = st.sidebar.selectbox(
    "Navigate",
    ["HOME", "DISEASE RECOGNITION", "PROJECT DETAILS"],
    index=["HOME", "DISEASE RECOGNITION", "PROJECT DETAILS"].index(st.session_state.page)
)

# --------------------
# Pages
# --------------------
if st.session_state.page == "HOME":
    st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 2rem;'>
            <h1 style='color: {text_color};'>Potato Leaf Disease Detection</h1>
            <p style='color: {text_color}; font-size: 18px;'>Upload a photo of a potato plant's leaf to detect any disease using a deep learning model.</p>
        </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "DISEASE RECOGNITION":
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image of a potato leaf", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = image.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = st.session_state.model.predict(img_array)
        class_names = ['Early Blight', 'Late Blight', 'Healthy']
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"Prediction: {predicted_class}")

elif st.session_state.page == "PROJECT DETAILS":
    st.subheader("About This Project")
    st.write("""
        This Streamlit app uses a deep learning model trained to identify common potato leaf diseases:

        - **Early Blight**
        - **Late Blight**
        - **Healthy**

        The model was built with TensorFlow and trained on a dataset of labeled images.
        Images are resized to 256x256, normalized, and passed through a CNN for classification.
    ""
    )
    
