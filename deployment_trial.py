import csv
import os
import pickle

import cv2
import face_recognition
import numpy as np


def preProcessImages():
    # This function goes in images folder and reads all the images and stores them in a list
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


def find_encodings(images, classNames, idNames):
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

def save_lists(list1, list2, filename):
    with open(filename, 'wb') as f:
        pickle.dump((list1, list2), f)

if __name__ == "__main__":
    # # trainModelOnNewPerson()
    image, names , ids= preProcessImages()
    print("Images Loaded Successfully...")
    encodeList, correspondingNames = find_encodings(images=image, classNames = names , idNames = ids)
    save_lists(encodeList, correspondingNames, 'encodings.pkl')
    print("encodings.pkl` file saved successfully...")
    # recognize_faces_in_video(encodeList, names)


# def recognize_faces_api(rgb_frame, saved_encodings, names):
#
#
#     # Find all face locations and encodings in the frame
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#     # Loop through each face found in the frame
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # Compare face encodings with saved encodings
#         matches = face_recognition.compare_faces(saved_encodings, face_encoding)
#         name = "Unknown"  # Default name if no match found
#
#         # Check for a match
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = names[first_match_index]  # Get the corresponding name
#
#         l1 = [top, right, left, bottom]
#         return jsonify({'personInImage':name, 'faceMarks':l1}), 200
    # d = {}
    # if 'image' not in request.files:
    #     return jsonify({'error': 'No image file provided'}), 400
    #
    # image_file = request.files['image']
    #
    # # Call the recognize_faces function with the image file
    # # recognized_faces = recognize_faces(image_file)
    #
    # img_1 = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
    # encode = face_recognition.face_encodings(img_1)[0]
    # d['encodings'] = encode
    # return jsonify({'recognized_faces': encode}), 200