from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("saved_model/final_model.h5")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Nutrient advice lookup
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

LABELS = {
    0: "Healthy",
    1: "Nitrogen Deficiency",
    2: "Phosphorus Deficiency",
    3: "Potassium Deficiency"
}

@app.route("/")
def home():
    return "Fertilizer Suggestion API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    if "crop" not in request.form:
        return jsonify({"error": "Please include 'crop' parameter."}), 400
    crop = request.form["crop"].lower()
    if crop not in {"rice","wheat","potato"}:
        return jsonify({"error": f"Crop '{crop}' not supported."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file part."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    fname = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(filepath)

    try:
        img = Image.open(filepath).resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)[0]
        idx = int(np.argmax(preds))
        label = LABELS.get(idx, "Unknown")
        confidence = float(round(np.max(preds), 2))
        solution = ADVICE.get(label, {}).get(crop, "No advice available for this crop.")

        return jsonify({
            "label": label,
            "confidence": confidence,
            "fertilizer_advice": solution
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # 👈 this is required for Render
    from waitress import serve
    serve(app, host='0.0.0.0', port=port)