# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Define function to load and preprocess image
def preprocess_image(image):
    # Resize image to match model input size
    image = image.resize((224, 224))
    # Convert image to array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define function to make predictions
def predict(image):
    # Preprocess image
    processed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(processed_image)
    return prediction

# Load your trained model
model = tf.keras.models.load_model('model21.h5')

# Define Streamlit app layout
def main():
    st.title('Plant Disease Detection App')
    st.sidebar.title('Options')

    # Upload multiple images
    uploaded_images = st.sidebar.file_uploader('Upload multiple images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_images:
        for idx, uploaded_image in enumerate(uploaded_images):
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption=f'Uploaded Image {idx+1}', use_column_width=True)

            # Make prediction when button is clicked
            if st.sidebar.button(f'Predict {idx+1}'):
                with st.spinner('Predicting...'):
                    prediction = predict(image)
                    if prediction[0][0] > 0.5:
                        st.success('The plant is predicted to be healthy!')
                    else:
                        st.error('The plant is predicted to be diseased!')

# Run the app
if __name__ == '__main__':
    main()
