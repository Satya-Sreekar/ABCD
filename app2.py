import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Function to load the model
def load_saved_model(model_path):
    model = load_model(model_path)
    return model

# Function to preprocess the image
def preprocess_image(image_data):
    img = image.load_img(image_data, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("Plant Disease Classification")
    st.sidebar.title("Model Settings")

    # Load the model
    model_path = "model21.h5"  # Change this to the path of your model file if not in the same directory
    if model_path is not None:
        model = load_saved_model(model_path)

        # Image upload for prediction
        st.subheader("Upload Image for Prediction")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            # Preprocess the image
            img_array = preprocess_image(uploaded_image)
            # Make prediction
            prediction = predict(model, img_array)
            # Output prediction
            if prediction[0][0] > 0.5:
                st.write("Prediction: Diseased")
            else:
                st.write("Prediction: Healthy")

if __name__ == "__main__":
    main()
