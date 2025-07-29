import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('saved_model/final_model.h5')

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save as TFLite file
with open("saved_model/model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as saved_model/model.tflite")
