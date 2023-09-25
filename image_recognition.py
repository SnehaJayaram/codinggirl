import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Create a function to perform image recognition and return the top predicted labels
def recognize_image(image_path):
    try:
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Adjust to match the input size of MobileNetV2
        img_array = np.array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Make predictions
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

        # Extract and return the top predicted labels
        top_labels = [label for (_, label, confidence) in decoded_predictions]
        return top_labels
    except Exception as e:
        return ["Error: " + str(e)]
      
