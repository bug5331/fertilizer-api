from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import gdown

# Constants
MODEL_ID = "1HqQVeFIYst7xGidIDEfHcXyLb8W6B_Yg"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "saved_model/final_model.h5"
UPLOAD_FOLDER = "uploads"

# Create necessary directories
os.makedirs("saved_model", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download the model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model using gdown...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully.")

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load the model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

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

# Routes
@app.route("/")
def home():
    return "✅ Fertilizer Suggestion API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    # Check crop input
    crop = request.form.get("crop", "").lower()
    if crop not in {"rice", "wheat", "potato"}:
        return jsonify({"error": "Invalid or missing 'crop'. Choose from rice, wheat, potato."}), 400

    # Check file input
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "Image file is missing."}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Preprocess image
        img = Image.open(filepath).resize((224, 224)).convert("RGB")
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict
        preds = model.predict(img_array)[0]
        idx = int(np.argmax(preds))
        label = LABELS.get(idx, "Unknown")
        confidence = round(float(np.max(preds)), 2)
        advice = ADVICE.get(label, {}).get(crop, "No advice available for this crop.")

        # Response
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

# Start the app using waitress for production
if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
