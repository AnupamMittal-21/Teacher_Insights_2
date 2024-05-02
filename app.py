import datetime
import pickle
import time
import dlib
import face_recognition
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)


def angle_between_points(p1, p2, p3):
    angle = abs(p2 - p1) - abs(p2 - p3)
    return angle


@app.route('/recognize_faces', methods=['POST', 'GET'])
def recognize_faces_api():
    if request.method == 'GET':
        return jsonify({'response': "Heyy GET request"}), 200

    elif request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        with open("encodings2.pkl", 'rb') as f:
            saved_encodings, names = pickle.load(f)
        print('Encoding File Loaded Successfully...')

        # Get the image file from the POST request
        image_file = request.files['image']

        # Read the image file and convert to image format that is supported by OpenCV
        image = face_recognition.load_image_file(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load the face detector and landmark predictor
        face_detector = dlib.get_frontal_face_detector()
        landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Threshold values: (HyperParameters)
        head_down_threshold = 3
        eye_aspect_ratio_threshold = 0.2

        # Variables to store the state of the student
        facesInfo = []
        Status = []

        # Detect faces in the image using the face detector and returns the rectangles of the faces (NOT USED)
        faces = face_detector(gray)

        # Find all face locations and encodings in the frame
        # Returns List of Faces in the form of Top, Right, Bottom, Left
        facesCurFrame = face_recognition.face_locations(image)
        encodesCurFrame = face_recognition.face_encodings(image, facesCurFrame)

        ts = time.time()
        frameDate = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        frameTime = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            # Compare the faces in the frame with the saved encodings
            matches = face_recognition.compare_faces(saved_encodings, encodeFace)
            faceDis = face_recognition.face_distance(saved_encodings, encodeFace)

            best_match_index = np.argmin(faceDis)
            name_ = names[best_match_index]

            confidence_threshold = 0.5

            faceInfo = {}
            if matches[best_match_index] and faceDis[best_match_index] < confidence_threshold:
                # Face recognized with confidence
                faceInfo['name'] = name_
                faceInfo['faceLoc'] = faceLoc
                faceInfo['confidence'] = 1 - faceDis[best_match_index]
            else:
                faceInfo['name'] = "UNKNOWN"
                faceInfo['faceLoc'] = None
                faceInfo['confidence'] = 0

            # Converting the face to a dlib rectangle
            top, right, bottom, left = faceLoc
            face = dlib.rectangle(left, top, right, bottom)
            landmarks = landmark_predictor(gray, face)

            # Extract the relevant facial landmarks (e.g., points around the eyes)
            left_eye_points = [landmarks.part(i) for i in range(36, 42)]
            right_eye_points = [landmarks.part(i) for i in range(42, 48)]

            # Calculate the angle between points to determine head orientation
            left_eye_angle = angle_between_points(left_eye_points[0].y, left_eye_points[1].y, left_eye_points[5].y)
            right_eye_angle = angle_between_points(right_eye_points[0].y, right_eye_points[1].y, right_eye_points[5].y)

            # Determine head orientation
            if left_eye_angle > head_down_threshold or right_eye_angle > head_down_threshold:
                head_status = "Head Down"
            else:
                head_status = "Head Up"

            vertical_dist_1 = ((left_eye_points[1].y - left_eye_points[5].y) ** 2 + (
                    left_eye_points[1].x - left_eye_points[5].x) ** 2) ** 0.5
            vertical_dist_2 = ((left_eye_points[2].y - left_eye_points[4].y) ** 2 + (
                    left_eye_points[2].x - left_eye_points[4].x) ** 2) ** 0.5
            horizontal_dist = ((left_eye_points[0].y - left_eye_points[3].y) ** 2 + (
                    left_eye_points[0].x - left_eye_points[3].x) ** 2) ** 0.5
            left_ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)

            vertical_dist_1 = ((right_eye_points[1].y - right_eye_points[5].y) ** 2 + (
                    right_eye_points[1].x - right_eye_points[5].x) ** 2) ** 0.5
            vertical_dist_2 = ((right_eye_points[2].y - right_eye_points[4].y) ** 2 + (
                    right_eye_points[2].x - right_eye_points[4].x) ** 2) ** 0.5
            horizontal_dist = ((right_eye_points[0].y - right_eye_points[3].y) ** 2 + (
                    right_eye_points[0].x - right_eye_points[3].x) ** 2) ** 0.5
            right_ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)

            # Determine eye status
            if left_ear < eye_aspect_ratio_threshold and right_ear < eye_aspect_ratio_threshold:
                eye_status = "Eyes Closed"
            else:
                eye_status = "Eyes Open"

            if head_status == "Head Up" and eye_status == "Eyes Open":
                attentiveness = "Attentive"
            else:
                attentiveness = "Not Attentive"

            faceInfo['HeadStatus'] = head_status
            faceInfo['EyeStatus'] = eye_status
            faceInfo['Attentiveness']= attentiveness
            facesInfo.append(faceInfo)

        return jsonify({'DateFrame':frameDate,'TimeFrame':frameTime,'Details of Attendance': facesInfo}), 200

@app.route('/check', methods=['POST', 'GET'])
def check():
    if request.method == 'GET':
        return jsonify({'response': "Heyy GET request"}), 200
    else:
        return jsonify({'response': "Heyy POST request"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
