import csv
import datetime
import os
import pickle
import time
from collections import deque

import cv2
import dlib
import face_recognition
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# This function goes in training images folder and reads all the images and stores them in a list
# and returns the list of images, classnames, and id names
def preProcessImages():
    path = 'TrainingImage'
    image = []
    idNames = []
    classNames = []
    mailNames = []
    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        image.append(curImg)
        classNames.append(os.path.split(cl)[1].split('.')[0])
        idNames.append(os.path.split(cl)[1].split('.')[1])
        mailNames.append(os.path.split(cl)[1].split('.')[2])
    return image, classNames, idNames

# returns the encodings of images and corresponding names from all the images passed in the function
def find_encodings(images, classNames):
    encode_list = []
    names = []
    for idx,img_1 in enumerate(images):
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img_1)
        if len(encode) == 0:
            print("No Face Detected in Image")
            continue
        encode = encode[0]
        names.append(classNames[idx])
        encode_list.append(encode)
    return encode_list, names

# Saves the encodings of new persons and corresponding images and corresponding names in a file
def trainModelOnNewPerson():
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()

    Id = '1'
    name = "Ace"

    if ((name.isalpha()) or (' ' in name)):
        cam = cv2.VideoCapture(0)
        cascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                cv2.imwrite("TrainingImage\ " + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                cv2.imshow('Taking Images', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 10:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Taken for ID : " + Id
        row = [serial, '', Id, '', name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
    else:
        if (name.isalpha() == False):
            res = "Enter Correct name"


# Opens camera and frame by frame recognises the face using saved_ecnodings that is returned by get_encoding function
# and displays the name of the person in the frame
def recognize_faces_in_video(saved_encodings, names):
    # Open video capture device (webcam)
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare face encodings with saved encodings
            matches = face_recognition.compare_faces(saved_encodings, face_encoding)
            name = "Unknown"  # Default name if no match found

            # Check for a match
            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]  # Get the corresponding name

            # Draw rectangle around the face and label it with the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # If 'q' is pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all windows
    video_capture.release()
    cv2.destroyAllWindows()


def angle_between_points(p1, p2, p3):
    angle = abs(p2 - p1) - abs(p2 - p3)
    return angle


def getHeadStatus(images, classNames):
    face_detector = dlib.get_frontal_face_detector()

    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    activity_history = deque(maxlen=100)
    head_down_threshold = 3
    eye_aspect_ratio_threshold = 0.2

    # File to store time-to-time scores
    score_file = open("student_scores.txt", "w")
    scoring_interval = 60  # e.g., score every 60 seconds
    timestamps = []
    scores = []
    # matches = []

    global best_match_index, matches, faceDis

    with open("encodings2.pkl", 'rb') as f:
        encode_list_known, names = pickle.load(f)
    print('Encoding File Loaded Successfully...')

    cap = cv2.VideoCapture(0)
    i = 0
    df = None
    attendance = None
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    attentiveness_data = pd.DataFrame(columns=['Timestamp', 'Attentiveness'])
    start_time = time.time()
    plot_open = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        head_status = "Undetermined"
        eye_status = "Undetermined"
        attentiveness = "Undetermined"

        # name of the preson
        faceLoc_ = None
        name_ = None
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encode_list_known, encodeFace)
            faceDis = face_recognition.face_distance(encode_list_known, encodeFace)

            best_match_index = np.argmin(faceDis)
            print(f"#######{classNames[best_match_index]} ##########")

            pat = str(classNames[best_match_index])
            # id1 = pat.split('.')[1]
            # id2 = id1.split('.')[0]

            print(f"#######{classNames[best_match_index]} ##########{1}##########")

            # Set a threshold for considering a match
            confidence_threshold = 0.5

            col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
            exists1 = os.path.isfile("StudentDetails\StudentDetails.csv")
            if exists1:
                df = pd.read_csv("StudentDetails\StudentDetails.csv")
            else:
                pass
            print(df.tail(3))

            if matches[best_match_index] and faceDis[best_match_index] < confidence_threshold:
                # Face recognized with confidence
                faceLoc_ = faceLoc
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                id2 = int(1)
                aa = df.loc[df['SERIAL NO.'] == id2]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == id2]['ID'].values
                ID = str(ID)
                # ID = ID[1:-1]
                print(f"ID IS {ID}")
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]

                recognized_name = "Person " + str(best_match_index + 1)
                print(f"Recognized: {recognized_name}, Confidence: {1 - faceDis[best_match_index]:.2%}")
                name = classNames[best_match_index].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back to the original size
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                name_ = name
                print(name)
            else:
                print("Face not recognized or confidence below threshold.")
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "UNKNOWN", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                # markAttendance(name)
        print("So i am now out of for loop, ")
        for face in faces:
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

            # Calculate eye aspect ratio (EAR)
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

        print("Out of for loop of head-UP")

        # Combine head and eye status to determine attentiveness
        if head_status == "Head Up" and eye_status == "Eyes Open":
            attentiveness = "Attentive"
        else:
            attentiveness = "Not Attentive"

        # Store attentiveness status for scoring
        activity_history.append(attentiveness)

        # Check if it's time to score
        elapsed_time = time.time() - start_time
        try:
            if elapsed_time >= scoring_interval:
                # Calculate the score based on the attentiveness history
                score = activity_history.count("Attentive") / len(activity_history)

                # Write the score to the file
                score_file.write(f"{time.ctime()}: {score}\n")

                # Store data for plotting
                timestamps.append(time.time())
                scores.append(score)

                attentiveness_data = attentiveness_data.append({'Timestamp': time.time(), 'Attentiveness': score},
                                                               ignore_index=True)

                # Reset variables for the next scoring interval
                activity_history.clear()
                start_time = time.time()
        except:
            print("Adding in file...")
        conf = 0.0

        rounded_score = 0.00

        if matches[best_match_index]:
            conf = faceDis[best_match_index]
            rounded_score = round(conf, 2)
        cv2.putText(frame, f"Head Status: {head_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {rounded_score}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Eye Status: {eye_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Attentiveness: {attentiveness}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Live Attention Monitoring", frame)

        try:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Check attentiveness on demand
                on_demand_score = activity_history.count("Attentive") / len(activity_history)
                print(f"On-demand Attentiveness: {on_demand_score * 100:.2f}%")
        except:
            print("Tapping q...")
            # Update the matplotlib plot dynamically
        if not plot_open:
            # Create a new matplotlib figure and axis
            fig, ax = plt.subplots()
            plot_open = True

        ax.clear()
        ax.plot(timestamps, scores, label='Attentiveness Score')
        ax.bar(timestamps, scores, label='Average Attentiveness Score', color='#81C4FF', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Attentiveness Score')
        ax.set_title('Class Attentiveness Over Time')
        ax.legend()

        # Display percentage activity on the graph
        if timestamps and scores:
            ax.text(timestamps[-1], scores[-1], f'{scores[-1] * 100:.2f}%', ha='right', va='bottom')

        # Draw the plot
        plt.draw()
        plt.pause(0.01)
    # Close the score file

    attentiveness_data.to_csv("attentiveness_data.csv", index=False)
    score_file.close()

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("After while loop bow ")

    date = datetime.datetime.now().strftime('%d-%m-%Y')

    exists = os.path.isfile("Attendance\Attendance_" + date + ".csv")
    if exists:
        print("in first if statement ")
        with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance)
        csvFile1.close()
    else:
        print("in first else statement ")
        with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
        csvFile1.close()
    # with open("Attendance\Attendance_" + date + ".csv", 'r') as csvFile1:
    #     reader1 = csv.reader(csvFile1)
    #     for lines in reader1:
    #         i = i + 1
    #         if (i > 1):
    #             if (i % 2 != 0):
    #                 iidd = str(lines[0]) + '   '
    #                 tv.insert('', 0, text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
    csvFile1.close()

    print("In function Track Images")

def save_lists(list1, list2, filename):
    with open(filename, 'wb') as f:
        pickle.dump((list1, list2), f)

if __name__ == "__main__":
    raw_images, image_names, ids = preProcessImages()
    # encodeList, correspondingNames = find_encodings(images=raw_images, classNames = image_names)
    # save_lists(encodeList, correspondingNames, 'encodings2.pkl')
    # print("encodings2.pkl` file saved successfully...")
    getHeadStatus(images=raw_images, classNames=image_names)
    # recognize_faces_in_video(encodeList, names)
