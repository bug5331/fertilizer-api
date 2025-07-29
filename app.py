from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown

# Constants
MODEL_ID = "1juIS2yzo8eeg3d62tSlA0AYdzlkC7IFX"  # Your Drive file ID
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "saved_model/model.tflite"
UPLOAD_FOLDER = "uploads"
os.makedirs("saved_model", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Download TFLite model from Google Drive if not exists
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("✅ Download complete.")

# Load the TFLite model
print("🔁 Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Model loaded.")

# Flask app setup
app = Flask(__name__)

# Class labels and fertilizer advice
LABELS = {
    0: "Healthy",
    1: "Nitrogen Deficiency",
    2: "Phosphorus Deficiency",
    3: "Potassium Deficiency"
}

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
        "rice": "No deficiency. Maintain balanced NPK fertilization.",
        "wheat": "No deficiency detected.",
        "potato": "No deficiency detected."
    }
}

@app.route("/")
def home():
    return "✅ Fertilizer Suggestion API (TFLite version) is live!"

@app.route("/predict", methods=["POST"])
def predict():
    crop = request.form.get("crop", "").lower()
    if crop not in {"rice", "wheat", "potato"}:
        return jsonify({"error": "Invalid or missing 'crop'. Choose from rice, wheat, potato."}), 400

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "Image file is missing."}), 400

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
        label = LABELS.get(idx, "Unknown")
        confidence = round(float(np.max(preds)), 2)
        advice = ADVICE.get(label, {}).get(crop, "No advice available for this crop.")

        return jsonify({
            "label": label,
            "confidence": confidence,
            "fertilizer_advice": advice
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# Production server
if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
