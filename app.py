import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# Page options
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION", "PROJECT DETAILS"])

# Theme detection inside render block
user_theme = st.runtime.scriptrunner.get_script_run_ctx().session_state.get("theme")
is_dark = user_theme and user_theme.get("base") == "dark"

# Title based on selected page
st.title("Potato Leaf Disease Detection")

# Load model lazily
@st.cache_resource
def load_model():
    model_path = "potato_disease_model.h5"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1nErjXKv6pN6WKnf8yqaV-x5VeV1XIKvP"
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

# Prediction function
def predict(image):
    model = load_model()
    image = image.resize((256, 256))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    prediction = model.predict(img_array)
    class_names = ['Early Blight', 'Late Blight', 'Healthy']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

# HOME PAGE
if app_mode == "HOME":
    st.subheader("Welcome to the Potato Leaf Disease Detection App")
    st.markdown("""
    This application uses a deep learning model to classify potato leaf diseases.

    **How to use:**
    1. Go to the **Disease Recognition** tab.
    2. Upload a potato leaf image.
    3. Get instant predictions with confidence.
    """)

# DISEASE RECOGNITION PAGE
elif app_mode == "DISEASE RECOGNITION":
    st.subheader("Upload a Potato Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button("Predict"):
            with st.spinner('Predicting...'):
                label, confidence = predict(image)
            st.success(f"Prediction: **{label}** with **{confidence * 100:.2f}%** confidence")

# PROJECT DETAILS PAGE
elif app_mode == "PROJECT DETAILS":
    st.subheader("About this Project")
    st.markdown("""
    - **Dataset**: Potato Leaf Disease Dataset
    - **Model**: Convolutional Neural Network (CNN) using TensorFlow
    - **Trained On**: Early Blight, Late Blight, and Healthy Leaf Images
    - **Features**: Real-time image classification, confidence score, responsive UI
    """)

# Footer styling based on theme
st.markdown("""
<style>
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if is_dark:
    st.markdown("""<style>body { background-color: #0E1117; color: white; }</style>""", unsafe_allow_html=True)
    
