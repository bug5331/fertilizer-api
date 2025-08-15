from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import sqlite3
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, initialize_app,db
import json
# -----------------------
# Constants & Setup
# -----------------------
MODEL_ID = "1juIS2yzo8eeg3d62tSlA0AYdzlkC7IFX"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "saved_model/model.tflite"
LABELS_PATH = "labels.txt"
UPLOAD_FOLDER = "uploads"
DB_PATH = "history.db"

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
SUGGESTIONS = {
    "Corn__Common_Rust": "Use a fungicide like Propiconazole. Rotate crops and remove infected debris.",
    "Corn__Gray_Leaf_Spot": "Apply fungicides such as strobilurins. Improve air circulation.",
    "Corn__Northern_Leaf_Blight": "Use nitrogen-rich fertilizer. Opt for resistant varieties.",
    "Potato__Early_Blight": "Apply copper-based fungicides. Use potassium nitrate and ensure proper irrigation.",
    "Potato__Late_Blight": "Use mancozeb-based fungicides. Avoid overhead irrigation.",
    "Rice__Brown_Spot": "Apply balanced NPK fertilizers. Improve drainage and use seed treatment.",
    "Rice__Leaf_Blast": "Apply potassium fertilizer. Avoid excess nitrogen.",
    "Rice__Neck_Blast": "Use tricyclazole fungicide. Maintain nitrogen levels properly.",
    "Sugarcane_Bacterial_Blight": "Apply copper oxychloride. Use disease-free setts and avoid injuries.",
    "Sugarcane_Red_Rot": "Apply phosphorus and potash. Use resistant varieties and rotate crops.",
    "Wheat__Brown_Rust": "Spray sulfur-based fungicides. Avoid dense planting.",
    "Wheat__Yellow_Rust": "Use Propiconazole. Apply nitrogen-based fertilizer.",
    "Corn__Healthy": "No fertilizer needed. Crop is healthy.",
    "Potato__Healthy": "No deficiency detected. Maintain balanced fertilization.",
    "Rice__Healthy": "No deficiency detected. Maintain balanced NPK fertilization.",
    "Sugarcane_Healthy": "No action required. Field is healthy.",
    "Wheat__Healthy": "No action needed. Crop is healthy."
}

# -----------------------
# Firebase Realtime DB
# -----------------------
cred_dict = json.loads(os.environ.get("FIREBASE_KEY_JSON"))
cred = credentials.Certificate(cred_dict)

initialize_app(cred, {
    'databaseURL': 'https://farmico-5c432-default-rtdb.firebaseio.com/'
})

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__)

# Init SQLite DB for history
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

# -----------------------
# Utilities
# -----------------------
def save_to_history(crop, condition, confidence=0.0):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO crop_tracker_history (crop, condition, confidence, timestamp) VALUES (?, ?, ?, ?)",
                   (crop, condition, confidence, timestamp))
    conn.commit()
    conn.close()

def preprocess_image(image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    return img_array

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return "✅ API running with prediction, history tracking, and CropTracker"

# -----------------------
# Prediction Endpoint
# -----------------------
@app.route("/predict", methods=["POST"])
def predict():
    crop = request.form.get("crop", "").lower()
    if crop not in {"rice", "wheat", "potato", "corn", "sugarcane"}:
        return jsonify({"status": "error", "message": "Invalid crop. Choose rice, wheat, potato, corn, sugarcane."}), 400

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"status": "error", "message": "Image file missing."}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img = Image.open(filepath)
        img_array = preprocess_image(img)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = int(np.argmax(preds))
        label = LABELS[idx]
        confidence = round(float(np.max(preds)) * 100, 2)
        suggestion = SUGGESTIONS.get(label, "No advice available.")
        save_to_history(crop, label, confidence)

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

# -----------------------
# Track Endpoint
# -----------------------
@app.route('/track', methods=['POST'])
def track():
    try:
        if 'image' not in request.files or request.files['image'].filename == "":
            return jsonify({'status': 'error', 'message': 'No image found'})

        image = Image.open(request.files['image'].stream)
        img_array = preprocess_image(image)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = int(np.argmax(preds))
        if idx < 0 or idx >= len(LABELS):
            return jsonify({'status': 'error', 'message': 'Prediction index out of range'})

        predicted_label = LABELS[idx]
        confidence = round(float(np.max(preds)) * 100, 2)

        if "__" in predicted_label:
            crop_name, condition = predicted_label.split("__")
        else:
            crop_name, condition = "Unknown", predicted_label

        save_to_history(crop_name, condition, confidence)

        return jsonify({
            'status': 'success',
            'prediction': {
                'label': predicted_label,
                'crop': crop_name,
                'condition': condition,
                'confidence_percent': f"{confidence}%"
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

# -----------------------
# CropTracker Endpoints
# -----------------------
@app.route("/crops", methods=["GET"])
def get_all_crops():
    try:
        crops_ref = db.reference("Crops")
        crops_data = crops_ref.get()
        if not crops_data:
            return jsonify({'status': 'error', 'message': 'No crops found'}), 404
        crop_names = list(crops_data.keys())
        return jsonify({'status': 'success', 'crops': crop_names})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/crops/<crop_name>", methods=["GET"])
def get_crop_details(crop_name):
    try:
        crops_ref = db.reference(f"Crops/{crop_name}")
        crop_data = crops_ref.get()
        if not crop_data:
            return jsonify({'status': 'error', 'message': f'Crop {crop_name} not found'}), 404
        return jsonify({'status': 'success', 'crop': crop_name, 'details': crop_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
import sqlite3
def get_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT crop, condition, confidence, timestamp FROM crop_tracker_history ORDER BY timestamp DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        history = [
            {
                "crop": row[0],
                "condition": row[1],
                "confidence": row[2],
                "timestamp": row[3]
            }
            for row in rows
        ]
        return jsonify({"status": "success", "data": history})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------
# Run with Waitress
# -----------------------
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
