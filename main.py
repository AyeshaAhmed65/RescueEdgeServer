from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import base64
import cv2
import numpy as np
import os
from datetime import datetime
import shutil
import mysql.connector

app = Flask(__name__)
CORS(app)

images_folder = "images"
predictions_folder = "Predictions"
os.makedirs(images_folder, exist_ok=True)
os.makedirs(predictions_folder, exist_ok=True)

# MySQL connection configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="rescueedge"
)
cursor = db.cursor()

def ai(name, gps_location):
    model = YOLO('humanFinder.pt')
    results = model.predict(f"./images/{name}", save=True, conf=0.3, classes=[0])
    for r in results:
        if len(r.boxes.xyxy) > 0:
            # Move image to 'Predictions' folder
            shutil.move(f"./runs/segment/predict/{name}", predictions_folder)

            # Insert image location and GPS location into the database
            cursor.execute("INSERT INTO detections (image_location, gps_location) VALUES (%s, %s)", (name, gps_location))
            db.commit()

    shutil.rmtree("./runs")


@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.get_json()
        img_base64 = data.get('image')  # Using .get() to handle missing 'image' key
        gps_location = data.get('gps_location')

        if img_base64 is None or gps_location is None:
            return jsonify({'status': 'error', 'message': 'Image data or GPS location missing in request.'}), 400

        img_bytes = base64.b64decode(img_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image data.'}), 400

        # Save the received image to the "images" folder with a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(images_folder, filename)
        cv2.imwrite(filepath, img)
        
        # Call ai() function with image filename and GPS location
        ai(filename, gps_location)

        return jsonify({'status': 'success', 'message': f'Image saved as {filename}'})

    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'message': str(e)}), 500  # Internal Server Error

# Route to serve static files (images) from the 'Predictions' folder
@app.route('/Predictions/<path:filename>')
def serve_static(filename):
    return send_from_directory(predictions_folder, filename)

@app.route('/Predictions')
def list_files():
    # Retrieve image location and GPS location from the database
    cursor.execute("SELECT image_location, gps_location FROM detections")
    results = cursor.fetchall()

    # Create a list of dictionaries containing image location and GPS location
    files = [{'image_location': row[0], 'gps_location': row[1]} for row in results]
    
    return jsonify(files)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5550)