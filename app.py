import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id ="1Dtc6aopehnUtOW78tpaTXGoaABwFjBo0"
url ='https://drive.google.com/uc?id=1Dtc6aopehnUtOW78tpaTXGoaABwFjBo0'
model_path ="trained_plant_disease_model.keras"


if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Check if model was downloaded correctly
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    st.success(f"Model downloaded successfully! File size: {file_size} bytes")
    
    if file_size < 89200000:  # If too small, it might be an HTML error file
        st.error("The downloaded file is too small. It might not be the correct model file.")
else:
    st.error("Model file was not found after download. Please check the URL or file permissions.")



def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page",["Home"," ","Disease Recognition"])

# import Image from pillow to open images
from PIL import Image
img = Image.open("Disease.png")

# display image using streamlit
# width is used to set the width of an image
st.image(img)

#Main Page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)

elif app_mode == "PROJECT DETAILS":
    st.header("Project Overview")
    st.write("""
    This project uses a **deep learning model** trained on potato leaf images to identify plant diseases with up to **93% accuracy**.
    
    **How It Works:**
    - The user uploads an image of a potato leaf.
    - The model processes the image and classifies it into three categories:
      1. **Potato Early Blight**
      2. **Potato Late Blight**
      3. **Healthy Potato Leaf**
    - The prediction is displayed on the screen.

    **Model Details:**
    - The model is built using **TensorFlow and Keras**.
    - It is trained on a dataset of potato leaves with different disease conditions.
    - The model uses **Convolutional Neural Networks (CNNs)** to analyze image patterns.

    **Why This Project?**
    - Helps farmers quickly detect diseases in potato crops.
    - Reduces the need for expert consultation.
    - Supports sustainable agriculture by early disease detection.
    """)
    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    st.markdown('This app detects the potato leaf disease with upto 93 percent accuracy')
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
