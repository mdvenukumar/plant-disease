import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('paddy_disease_model1.h5')

# Function to preprocess the image
def preprocess_image(image):
    resized_image = image.resize((256, 256))
    img_array = np.array(resized_image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction

# UI
st.set_page_config(page_title="Paddy Disease Classification App", page_icon=":herb:")

# Title and description
st.title('Paddy Disease Classification App')
st.write("Upload an image of a paddy leaf and let the model predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Predictions
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            prediction = predict(image)
            # Map predicted class index to disease name
            class_mapping = {
                0: "Normal",
                1: "Dead Heart",
                2: "Brown Spot",
            }
            predicted_class_index = np.argmax(prediction)
            predicted_disease = class_mapping.get(predicted_class_index, "Unknown")
            # Get the confidence score
            confidence_score = prediction[0][predicted_class_index]

            # Display the prediction results above the image
            col1, col2 = st.columns([2, 3])
            with col1:
                st.subheader("Prediction Results:")
                st.write(f"Predicted Disease: **{predicted_disease}**")
                st.write(f"Confidence: {confidence_score:.2f}")
            with col2:
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
            # Celebratory balloons for correct prediction
            if predicted_disease != "Unknown":
                st.balloons()
else:
    st.info("Please upload an image.")
hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
