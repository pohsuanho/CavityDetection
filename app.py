import tensorflow as tf
import cv2
import streamlit as st
import numpy as np

model_directory = 'model'
loaded_model = tf.saved_model.load(model_directory)

def load_image(image_file):
    # Read image file with OpenCV
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_resized = cv2.resize(image, (120, 120))   # Resize images for model
    image_preprocessed = tf.keras.applications.vgg16.preprocess_input(image_resized) 
    return image, image_preprocessed

def visualize_predicted_image_opencv(image, coordinates):
    height, width, _ = image.shape

    # Denormalize coordinates
    x_min, y_min, x_max, y_max = coordinates
    x_min_denorm = int(x_min * width)
    y_min_denorm = int(y_min * height)
    x_max_denorm = int(x_max * width)
    y_max_denorm = int(y_max * height)

    # Draw bounding box
    cv2.rectangle(image, (x_min_denorm, y_min_denorm), (x_max_denorm, y_max_denorm), (255, 0, 0), 2)

    return image

st.title('Upload your X-ray here')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image, preprocessed_image = load_image(uploaded_file)
    image_batch = np.expand_dims(preprocessed_image, axis=0)

    # Use the loaded model for prediction
    inference_function = loaded_model.signatures['serving_default']
    predictions = inference_function(tf.constant(image_batch, dtype=tf.float32))

    # Extract and process predictions
    classification_pred = predictions['classification_output'].numpy()
    regression_pred = predictions['regression_output'].numpy()

    # Visualization
    coordinates = regression_pred[0]
    result_image = visualize_predicted_image_opencv(image, coordinates)

    # Display
    st.image(result_image, caption='Processed Image', use_column_width=True)














