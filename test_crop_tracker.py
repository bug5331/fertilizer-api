import requests

# API URL
url = "https://fertilizer-api-ylvo.onrender.com/predict"

# Path to your image
image_path = r"C:\Users\bhagw\OneDrive\Desktop\IMG20201109164156_00.jpg"

# Data to send
data = {
    "crop": "wheat"  # or any other crop like "rice", "corn", etc.
}

# Open image in binary mode
files = {
    "file": open(image_path, "rb")
}

# Send POST request
response = requests.post(url, data=data, files=files)

# Print the response
print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Response Text:", response.text)
