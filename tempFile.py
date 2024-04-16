import threading

import cv2, os
from collections import deque
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import face_recognition
import dlib
import matplotlib

matplotlib.use('TkAgg')

path = 'TrainingImage'
image = []
idNames = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    image.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    idNames.append(os.path.splitext(cl)[1])
print(classNames)


#######################################################################################

# using the haar cascade for detecting the face and then doing the singUP
def TakeImages():
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


########################################################################################
def find_encodings(images):
    encode_list = []
    cnt = 0
    for img_1 in images:
        try:
            cnt += 1
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img_1)[0]
            encode_list.append(encode)
        except:
            print("Processing...")
    return encode_list


face_detector = dlib.get_frontal_face_detector()

landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# When we click on "take Attendance" then it is called
def TrainImages():
    print('Encoding Complete')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImage")
    try:
        recognizer.train(faces, np.array(ID))
    except:
        return
    recognizer.save("TrainingImageLabel\Trainner.yml")


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids


###########################################################################################

def TrackImages():
    global best_match_index, matches, faceDis
    encode_list_known = find_encodings(image)
    print('Encoding Complete')

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
            print(f"#######{image[best_match_index]} ##########")

            pat = str(classNames[best_match_index])
            id1 = pat.split('.')[1]
            id2 = id1.split('.')[0]

            print(f"#######{classNames[best_match_index]} ##########{id2}##########")

            # Set a threshold for considering a match
            confidence_threshold = 0.5

            col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
            exists1 = os.path.isfile("StudentDetails\StudentDetails.csv")
            if exists1:
                df = pd.read_csv("StudentDetails\StudentDetails.csv")
            else:
                print("No file")

            print(df.tail(3))

            if matches[best_match_index] and faceDis[best_match_index] < confidence_threshold:
                # Face recognized with confidence
                faceLoc_ = faceLoc
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                id2 = int(id2)
                aa = df.loc[df['SERIAL NO.'] == id2]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == id2]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
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

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("After while loop bow ")
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
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
    with open("Attendance\Attendance_" + date + ".csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if (i > 1):
                if (i % 2 != 0):
                    iidd = str(lines[0]) + '   '
                    print(iidd)
    csvFile1.close()

    print("In function Track Images")


def recognize_faces_in_video(saved_encodings, names):
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []

    def detect_faces():
        nonlocal face_locations, face_encodings
        while True:
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and encodings in the frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Start face detection thread
    face_detection_thread = threading.Thread(target=detect_faces)
    face_detection_thread.start()

    while True:
        ret, frame = video_capture.read()

        # Scale up face locations since the frame was resized
        face_locations_scaled = [(top * 4, right * 4, bottom * 4, left * 4) for (top, right, bottom, left) in
                                 face_locations]

        # Draw rectangles and labels on the frame using face locations and names
        for (top, right, bottom, left), face_encoding in zip(face_locations_scaled, face_encodings):
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

        cv2.imshow('Video', frame)

        # If 'q' is pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all windows
    video_capture.release()
    cv2.destroyAllWindows()