import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Preprocess uploaded image
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Load trained model
def load_trained_model(model_path):
    return load_model(model_path)

# Predict class
def predict_disease(model, img_array):
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])

    # Add your actual class names here
    class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Healthy']
    return class_names[class_idx]
