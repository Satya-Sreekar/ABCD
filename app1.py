import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('model21.h5')

# Define function to preprocess multiple images
def preprocess_images(uploaded_images):
    processed_images = []
    for uploaded_image in uploaded_images:
        image = Image.open(uploaded_image)
        # Resize image to match model input size
        image = image.resize((224, 224))
        # Convert image to array and normalize
        image_array = np.array(image) / 255.0
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        processed_images.append(image_array)
    return processed_images

# Define function to make predictions
def predict(images):
    predictions = []
    for image in images:
        # Make prediction
        prediction = model.predict(image)
        (healthy, disease) = prediction[0]
        class_names=['healthy', 'disease']
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        return predicted_class, prediction[0][predicted_class_index]

# Define Streamlit app layout
def main():
    st.title('Plant Disease Detection App')
    st.sidebar.title('Options')

    # Upload multiple images
    uploaded_images = st.sidebar.file_uploader('Upload multiple images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_images:
        processed_images = preprocess_images(uploaded_images)
        for idx, processed_image in enumerate(processed_images):
            # Display uploaded image
            st.image(Image.open(uploaded_images[idx]), caption=f'Uploaded Image {idx+1}', use_column_width=True)

            # Make prediction when button is clicked
            if st.sidebar.button(f'Predict {idx+1}'):
                with st.spinner('Predicting...'):
                    predicted_class, accuracy = predict([processed_image])

                    # Display prediction result
                    if accuracy > 0.5:
                        st.success(f'Image {idx+1}: Predicted as {predicted_class} with {accuracy:.2%} confidence')
                    else:
                        st.error(f'Image {idx+1}: Predicted as {predicted_class} with {accuracy:.2%} confidence')

# Run the app
if __name__ == '__main__':
    main()
