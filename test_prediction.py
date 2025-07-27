import requests

# 🔗 URL of your local Flask API
url = "http://127.0.0.1:5000/predict"  # Or use "http://192.168.x.x:5000/predict" if testing from Android

# 📷 Path to a test image (must exist in the same folder or give full path)
file_path = r"C:\Users\user\OneDrive\Desktop\bhagwan\PROJECT\backend\uploads\image_99.JPG"  # Change this to match the image file you want to test

# 🌾 Choose crop: "rice", "wheat", or "potato"
crop = "rice"

# 📤 Send POST request with image and crop name
with open(file_path, "rb") as img:
    files = {"file": img}
    data = {"crop": crop}
    response = requests.post(url, files=files, data=data)

# 📥 Print server response
try:
    print("✅ Server Response:")
    print(response.json())
except Exception as e:
    print("❌ Error:", e)
