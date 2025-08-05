from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model (same for both features)
model = load_model("saved_model/final_model.h5")

# Labels for fertilizer prediction
with open("labels.txt", "r") as f:
    fertilizer_labels = [line.strip() for line in f.readlines()]

# Labels for crop + disease prediction
crop_disease_labels = [
    "Corn__Common_Rust", "Corn__Gray_Leaf_Spot", "Corn__Healthy", "Corn__Northern_Leaf_Blight",
    "Potato__Early_Blight", "Potato__Healthy", "Potato__Late_Blight",
    "Rice__Brown_Spot", "Rice__Healthy", "Rice__Leaf_Blast", "Rice__Neck_Blast",
    "Sugarcane__Bacterial_Blight", "Sugarcane__Healthy", "Sugarcane__Red_Rot",
    "Wheat__Brown_Rust", "Wheat__Healthy", "Wheat__Yellow_Rust"
]

fertilizer_suggestions = {
    "Nitrogen Deficiency": "Apply Urea or Ammonium Sulfate.",
    "Phosphorus Deficiency": "Use Single Super Phosphate (SSP) or DAP.",
    "Potassium Deficiency": "Apply Muriate of Potash (MOP).",
    "Healthy": "Your crop is healthy. No fertilizer needed."
}

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # Convert RGBA to RGB if needed
        image = image[:, :, :3]
    return np.expand_dims(image, axis=0)

@app.route("/predict-fertilizer", methods=["POST"])
def predict_fertilizer():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files['image'].read()
    processed = preprocess_image(image_bytes)
    prediction = model.predict(processed)
    index = np.argmax(prediction)
    label = fertilizer_labels[index]
    suggestion = fertilizer_suggestions.get(label, "No suggestion available.")

    return jsonify({
        "label": label,
        "suggestion": suggestion
    })

@app.route("/predict-crop-disease", methods=["POST"])
def predict_crop_disease():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files['image'].read()
    processed = preprocess_image(image_bytes)
    prediction = model.predict(processed)
    index = np.argmax(prediction)
    label = crop_disease_labels[index]

    crop, disease = label.split("__")

    return jsonify({
        "crop": crop,
        "disease": disease
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown

# Constants
MODEL_ID = "1juIS2yzo8eeg3d62tSlA0AYdzlkC7IFX"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "saved_model/model.tflite"
LABELS_PATH = "labels.txt"
UPLOAD_FOLDER = "uploads"

os.makedirs("saved_model", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("✅ Download complete.")

# Load TFLite model
print("🔁 Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Model loaded.")

# Load labels from labels.txt in root
def load_labels(path=LABELS_PATH):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

LABELS = load_labels()

# Advice mapping
ADVICE = {
    "Nitrogen Deficiency": {
        "rice": "Apply Urea @ 60 kg/acre after tillering",
        "wheat": "Apply Urea @ 80 kg/acre at early vegetative stage",
        "potato": "Top-dress Urea @ 60 kg/acre 3–4 weeks after planting"
    },
    "Phosphorus Deficiency": {
        "rice": "Apply Single Super Phosphate @ 40 kg/acre at planting",
        "wheat": "Apply DAP @ 50 kg/acre at sowing",
        "potato": "Apply SSP @ 50 kg/acre before planting"
    },
    "Potassium Deficiency": {
        "rice": "Apply MOP @ 30 kg/acre at panicle initiation",
        "wheat": "Apply SOP @ 40 kg/acre at tillering stage",
        "potato": "Apply MOP @ 40 kg/acre at hilling stage"
    },
    "Healthy": {
        "rice": "No deficiency detected. Maintain balanced NPK fertilization.",
        "wheat": "No deficiency detected.",
        "potato": "No deficiency detected."
    }
}

# Flask app setup
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Fertilizer Suggestion + Deficiency Detection API (TFLite) is live!"

@app.route("/predict", methods=["POST"])
def predict():
    crop = request.form.get("crop", "").lower()
    if crop not in {"rice", "wheat", "potato"}:
        return jsonify({
            "status": "error",
            "message": "❌ Invalid or missing crop. Choose from: rice, wheat, potato."
        }), 400

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({
            "status": "error",
            "message": "❌ No image file provided."
        }), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img = Image.open(filepath).resize((224, 224)).convert("RGB")
        img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = int(np.argmax(preds))
        label = LABELS[idx]
        confidence = round(float(np.max(preds)) * 100, 2)

        # Fertilizer suggestion
        advice = ADVICE.get(label, {}).get(crop, "No specific advice available.")

        return jsonify({
            "status": "success",
            "prediction": {
                "condition": label,
                "confidence_percent": f"{confidence}%",
                "crop": crop.capitalize(),
                "fertilizer_suggestion": advice
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"⚠️ Server error: {str(e)}"
        }), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# For local or render deployment
if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
