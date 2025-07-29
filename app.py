import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Constants
model_url = "https://drive.google.com/uc?id=1Dtc6aopehnUtOW78tpaTXGoaABwFjBo0"
model_path = "trained_plant_disease_model.keras"

# Download model if not exists
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# Verify model
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    st.success(f"Model downloaded successfully! File size: {file_size / 1_000_000:.2f} MB")
    if file_size < 89_000_000:
        st.error("Downloaded file seems too small. Possibly incorrect.")
else:
    st.error("Model download failed. Please check the link or file permissions.")

# Prediction function
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar navigation
st.sidebar.title("🌿 Plant Disease Detection")
app_mode = st.sidebar.radio("Navigate", ["🏠 HOME", "🔬 DISEASE RECOGNITION", "📊 PROJECT DETAILS"])

# Banner Image
st.image("Disease.png", use_container_width=True)

# HOME
if app_mode == "🏠 HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: blue;'>👉 Select 'Disease Recognition' from the sidebar to get started! 👈</h3>", unsafe_allow_html=True)

    moving_link = """
    <style>
        .glow {
            font-size: 20px;
            color: #00FFFF;
            text-align: center;
            animation: glow-effect 1s infinite alternate;
        }
        @keyframes glow-effect {
            from {text-shadow: 0 0 5px #00FFFF, 0 0 10px #00FFFF;}
            to {text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF;}
        }
    </style>
    <marquee behavior="scroll" direction="left" scrollamount="5">
        <span class="glow">✨</span> 
        <a href="https://github.com/Rushikesav/Test-data/tree/main/3.Potato%20Leaf%20Disease%20Detection/dataset/Test" target="_blank" style="text-decoration: none; color: blue;">
            Click here to download the test data to test the model! 🌿
        </a> 
        <span class="glow">✨</span>
    </marquee>
    """
    st.markdown(moving_link, unsafe_allow_html=True)

# DISEASE RECOGNITION
elif app_mode == "🔬 DISEASE RECOGNITION":
    st.header("Upload a Potato Leaf Image")
    st.markdown('This model detects potato leaf disease with up to **93% accuracy**.')
    test_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if st.button("Show Image"):
        if test_image:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image.")

    if st.button("Predict"):
        if test_image:
            st.snow()
            st.write("🔍 **Prediction Result:**")
            index = model_prediction(test_image)
            class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success(f"**Model predicts:** {class_names[index]}")
        else:
            st.warning("Please upload an image.")

# PROJECT DETAILS
elif app_mode == "📊 PROJECT DETAILS":
    st.title("Project Details and Model Working")

    st.header("1. Introduction")
    st.markdown("""
    This is an AI-powered system to detect potato leaf diseases. It helps farmers make informed decisions using:
    - **Early Blight**
    - **Late Blight**
    - **Healthy leaf** classification.
    """)

    st.header("2. How It Works")
    st.markdown("""
    - Uses **Convolutional Neural Networks (CNN)**
    - Images resized to **128x128**
    - Highest predicted class selected from 3 categories
    """)

    st.header("3. Tech Stack")
    st.markdown("""
    - **TensorFlow/Keras**: Deep Learning Model
    - **Streamlit**: Web Interface
    - **gdown**: For Google Drive model download
    """)

    st.header("4. Future Scope")
    st.markdown("""
    - Add more crops and diseases
    - Mobile and drone integration
    - Real-time camera input
    """)

