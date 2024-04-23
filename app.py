import pickle

import face_recognition
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import numpy as np
import io

app = Flask(__name__)
camera = cv2.VideoCapture(0)


@app.route('/recognize_faces', methods=['POST', 'GET'])
def recognize_faces_api():
    if request.method == 'GET':
        return jsonify({'response': "Heyy GET request"}), 200
    elif request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']

        # Read the image file
        image = face_recognition.load_image_file(image_file)

        with open("encodings.pkl", 'rb') as f:
            saved_encodings, names = pickle.load(f)

        # Find all face locations and encodings in the frame
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Loop through each face found in the frame
        top = 0
        right = 0
        bottom = 0
        left = 0
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare face encodings with saved encodings
            matches = face_recognition.compare_faces(saved_encodings, face_encoding)
            name = "Unknown"  # Default name if no match found

            # Check for a match
            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]  # Get the corresponding name

            l1 = [top, right, left, bottom]
            return jsonify({'personInImage': name, 'faceMarks': l1}), 200

        return jsonify({'personInImage': 'Unknown', 'faceMarks': [top, right, left, bottom]}), 200

@app.route('/trial', methods=['POST', 'GET'])
def gettt():
    if request.method == 'GET':
        return jsonify({'response': "Heyy GET request"}), 200
    else :
        return jsonify({'response': "Heyy POST request"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
