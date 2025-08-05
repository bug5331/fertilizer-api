from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import sqlite3
from datetime import datetime

# Constants
MODEL_ID = "1juIS2yzo8eeg3d62tSlA0AYdzlkC7IFX"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "saved_model/model.tflite"
LABELS_PATH = "labels.txt"
UPLOAD_FOLDER = "uploads"
DB_PATH = "history.db"

# Setup
os.makedirs("saved_model", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
def load_labels(path=LABELS_PATH):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]
LABELS = load_labels()

# Fertilizer suggestions
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

# Setup Flask app
app = Flask(__name__)

# Init DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS crop_tracker_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        crop TEXT NOT NULL,
        condition TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return "✅ API running with prediction and history tracking"

@app.route("/predict", methods=["POST"])
def predict():
    crop = request.form.get("crop", "").lower()
    if crop not in {"rice", "wheat", "potato"}:
        return jsonify({"status": "error", "message": "Invalid crop. Choose rice, wheat, potato."}), 400

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"status": "error", "message": "Image file missing."}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Preprocess image
        img = Image.open(filepath).resize((224, 224)).convert("RGB")
        img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        # Predict
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = int(np.argmax(preds))
        label = LABELS[idx]
        confidence = round(float(np.max(preds)) * 100, 2)
        suggestion = ADVICE.get(label, {}).get(crop, "No advice available.")

        # Save to DB
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO crop_tracker_history (crop, condition, confidence, timestamp) VALUES (?, ?, ?, ?)",
                       (crop, label, confidence, timestamp))
        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "prediction": {
                "crop": crop.capitalize(),
                "condition": label,
                "confidence_percent": f"{confidence}%",
                "fertilizer_suggestion": suggestion
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            
@app.route('/track', methods=['POST'])
def track():
    try:
        image = request.files['image']
        if image:
            image = Image.open(image.stream)
            img_array = preprocess_image(image)

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.argmax(output_data, axis=1)

            if prediction is not None and 0 <= prediction[0] < len(labels):
                predicted_label = labels[prediction[0]]
            else:
                return jsonify({'status': 'error', 'message': 'Prediction index out of range'})

            # Extract crop and condition
            if '__' in predicted_label:
                crop_name, condition = predicted_label.split('__')
            else:
                crop_name, condition = "Unknown", predicted_label

            # Save to DB
            save_to_history(crop_name, condition)

            return jsonify({'status': 'success', 'prediction': predicted_label})
        else:
            return jsonify({'status': 'error', 'message': 'No image found'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'})


# Use waitress for production
if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
