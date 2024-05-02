############################################# IMPORTING ################################################
import pickle
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
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
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.offline as pyo
import matplotlib

matplotlib.use('TkAgg')


############################################# FUNCTIONS #################################

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


##################################################################################

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)


###################################################################################

def contact():
    mess._show(title='Contact us', message="Please contact us on : 'teacherinsights@gmail.com' ")


###################################################################################

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()


###################################################################################

def save_pass():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    op = (old.get())
    newp = (new.get())
    nnewp = (nnew.get())
    if (op == key):
        if (newp == nnewp):
            txf = open("TrainingImageLabel\psd.txt", "w")
            txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()


###################################################################################

def change_pass():
    global master
    master = tk.Tk()
    master.geometry("400x160")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")
    lbl4 = tk.Label(master, text='    Enter Old Password', bg='white', font=('times', 12, ' bold '))
    lbl4.place(x=10, y=10)
    global old
    old = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    old.place(x=180, y=10)
    lbl5 = tk.Label(master, text='   Enter New Password', bg='white', font=('times', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    new.place(x=180, y=45)
    lbl6 = tk.Label(master, text='Confirm New Password', bg='white', font=('times', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    nnew.place(x=180, y=80)
    cancel = tk.Button(master, text="Cancel", command=master.destroy, fg="black", bg="red", height=1, width=25,
                       activebackground="white", font=('times', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#3ece48", height=1, width=25,
                      activebackground="white", font=('times', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()


#####################################################################################

def psw():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        key = tf.read()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if (password == key):
        TrainImages()
    elif (password == None):
        pass
    else:
        mess._show(title='Wrong Password', message='You have entered wrong password')


######################################################################################

def clear():
    txt.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)


##########################################################################


def clear2():
    txt2.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)


#######################################################################################

def trainModelOnNewPerson():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        # Floor division (%) operator
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()

    Id = (txt.get())
    name = (txt2.get())

    if ((name.isalpha()) or (' ' in name)):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
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
            elif sampleNum > 50:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Taken for ID : " + Id
        row = [serial, '', Id, '', name]

        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
    else:
        if (name.isalpha() == False):
            res = "Enter Correct name"
            message.configure(text=res)


##########################################################################


def angle_between_points(p1, p2, p3):
    angle = abs(p2 - p1) - abs(p2 - p3)
    return angle


##########################################################################


def TrainImages():
    print('Encoding Complete')

    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")

    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImage")
    try:
        recognizer.train(faces, np.array(ID))
    except:
        mess._show(title='No Registrations', message='Please Register someone first!!!')
        return
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Profile Saved Successfully"
    message1.configure(text=res)
    message.configure(text='Total Registrations till now  : ' + str(ID[0]))

    if os.path.exists('data.pickle'):
        with open('data.pickle', 'wb') as f:  # 'wb' for binary write mode
            # Clear the file by writing an empty byte string
            f.write(b'')
    else:
        open('data.pickle', 'x').close()  # Create a new empty file if it doesn't exist

    # Perform your data processing
    images, classNames, idNames = preProcessImages()
    saved_encodings, names, ids = find_encodings(images, classNames, idNames)

    # Create data dictionary
    data = {
        'saved_encodings': saved_encodings,
        'names': names,
        'ids': ids
    }

    # Write the data to the file
    with open('data.pickle', 'wb') as f:  # 'wb' for binary write mode
        pickle.dump(data, f)


##########################################################################

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empty face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
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


############################################################################################


##########################################################################


##########################################################################


##########################################################################


##########################################################################


##########################################################################


##########################################################################


##########################################################################


##########################################################################


##########################################################################


##########################################################################


##########################################################################


##########################################################################
def mainFun():
    if os.path.exists('data.pickle'):
        with open('data.pickle', 'rb') as f:
            loaded_data = pickle.load(f)

        saved_encodings = loaded_data['saved_encodings']
        names = loaded_data['names']
        ids = loaded_data['ids']

    else:
        images, classNames, idNames = preProcessImages()
        saved_encodings, names, ids = find_encodings(images, classNames, idNames)

        data = {
            'saved_encodings': saved_encodings,
            'names': names,
            'ids': ids
        }

        with open('data.pickle', 'wb') as f:
            pickle.dump(data, f)

    recognize_faces_api(saved_encodings, names)


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


def find_encodings(images, classNames, idNames):
    encode_list = []
    names = []
    ids = []
    for idx, img_1 in enumerate(images):
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img_1)
        if len(encode) == 0:
            print("No Face Detected in Image")
            continue
        encode = encode[0]
        names.append(classNames[idx])
        ids.append(idNames[idx])
        encode_list.append(encode)
    return encode_list, names, ids


########################################################################################

# This is to test, this is a basic code and nothing else.
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


###########################################################################################


def recognize_faces_api(saved_encodings, names):

    # Load the face detector and landmark predictor
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Threshold values: (HyperParameters)
    head_down_threshold = 3
    eye_aspect_ratio_threshold = 0.2

    # Variables to store the state of the student
    facesInfo = []
    Status = []

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image using the face detector and returns the rectangles of the faces (NOT USED)
        faces = face_detector(gray)

        # Find all face locations and encodings in the frame
        # Returns List of Faces in the form of Top, Right, Bottom, Left
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

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
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            print(faceLoc)

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
            faceInfo['Attentiveness'] = attentiveness
            cv2.putText(frame, faceInfo['HeadStatus'], (x1 + 6, y2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, faceInfo['EyeStatus'], (x1 + 6, y2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, faceInfo['Attentiveness'], (x1 + 6, y2 - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, faceInfo['name'], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),1)
            facesInfo.append(faceInfo)

        cv2.imshow('Live Class Monitoring', frame)
        try:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Pressing c...")
                # Check attentiveness on demand
                # on_demand_score = activity_history.count("Attentive") / len(activity_history)
                # print(f"On-demand Attentiveness: {on_demand_score * 100:.2f}%")
        except:
            print("Tapping q...")

    cap.release()
    cv2.destroyAllWindows()

######################################## USED STUFFS ############################################

global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

mont = {'01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
        }

######################################## GUI FRONT-END ###########################################


window = tk.Tk()
window.geometry("1960x1080")
window.resizable(True, False)
window.title("Attendance System")
window.configure(background='#262523')

frame1 = tk.Frame(window, bg="#00aeff")
frame1.place(relx=0.06, rely=0.16, relwidth=0.43, relheight=0.60)

frame2 = tk.Frame(window, bg="#00aeff")
frame2.place(relx=0.51, rely=0.16, relwidth=0.43, relheight=0.60)

message3 = tk.Label(window, text="Teacher Insights", fg="white", bg="#262523", width=70,
                    height=1, font=('times', 29, ' bold '))
message3.place(x=40, y=9)

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.51, rely=0.08, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.33, rely=0.08, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4, text=day + "-" + mont[month] + "-" + year, fg="orange", bg="#262523", width=55,
                 height=1, font=('times', 22, ' bold '))
datef.pack(fill='both', expand=1)

clock = tk.Label(frame3, fg="orange", bg="#262523", width=55, height=1, font=('times', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

head2 = tk.Label(frame2,
                 text="                                             For New Registrations                                              ",
                 fg="black",
                 bg="#3ece48", font=('times', 17, ' bold '))
head2.grid(row=0, column=0)

head1 = tk.Label(frame1,
                 text="                                         For Already Registered                                                 ",
                 fg="black",
                 bg="#3ece48", font=('times', 17, ' bold '))
head1.place(x=0, y=0)

lbl = tk.Label(frame2, text="Enter ID", width=25, height=1, fg="black", bg="#00aeff", font=('times', 17, ' bold '))
lbl.place(x=5, y=55)

txt = tk.Entry(frame2, width=40, fg="black", font=('times', 15, ' bold '))
txt.place(x=140, y=88)

lbl2 = tk.Label(frame2, text="Enter Name", width=25, fg="black", bg="#00aeff", font=('times', 17, ' bold '))
lbl2.place(x=20, y=140)

txt2 = tk.Entry(frame2, width=40, fg="black", font=('times', 15, ' bold '))
txt2.place(x=140, y=173)

message1 = tk.Label(frame2, text="Step 1) : Take Images  then 2): Save Profile", bg="#00aeff", fg="black", width=50,
                    height=1,
                    activebackground="yellow", font=('times', 15, ' bold '))
message1.place(x=25, y=230)

message = tk.Label(frame2, text="", bg="#00aeff", fg="black", width=39, height=1, activebackground="yellow",
                   font=('times', 16, ' bold '))
message.place(x=40, y=450)

lbl3 = tk.Label(frame1, text="Attendance", width=20, fg="black", bg="#00aeff", height=1, font=('times', 17, ' bold '))
lbl3.place(x=100, y=115)

res = 0
exists = os.path.isfile("StudentDetails\StudentDetails.csv")
if exists:
    with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
    csvFile1.close()
else:
    res = 0
message.configure(text='Total Registrations till now  : ' + str(res))

##################### MENUBAR #################################

menubar = tk.Menu(window, relief='ridge')
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='Help', font=('times', 29, ' bold '), menu=filemenu)

################## TREEVIEW ATTENDANCE TABLE ####################

tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=82)
tv.column('name', width=130)
tv.column('date', width=133)
tv.column('time', width=133)
tv.grid(row=2, column=0, padx=(80, 0), pady=(150, 0), columnspan=4)
tv.heading('#0', text='ID')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

###################### SCROLLBAR ################################

scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 0), pady=(150, 0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

###################### BUTTONS ##################################

clearButton = tk.Button(frame2, text="Clear", command=clear, fg="black", bg="#ea2a2a", width=11,
                        activebackground="white", font=('times', 11, ' bold '))
clearButton.place(x=450, y=86)
clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="black", bg="#ea2a2a", width=11,
                         activebackground="white", font=('times', 11, ' bold '))
clearButton2.place(x=450, y=172)
takeImg = tk.Button(frame2, text="Take Images", command=trainModelOnNewPerson, fg="white", bg="blue", width=34,
                    height=1,
                    activebackground="white", font=('times', 15, ' bold '))
takeImg.place(x=142, y=300)
trainImg = tk.Button(frame2, text="Save Profile", command=psw, fg="white", bg="blue", width=34, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trainImg.place(x=142, y=380)
trackImg = tk.Button(frame1, text="Take Attendance", command=mainFun, fg="black", bg="yellow", width=35, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trackImg.place(x=142, y=50)

# igraph = tk.Button(frame1, text="Individual Graph", command=window.destroy  ,fg="black"  ,bg="red"  ,width=15 ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
# igraph.place(x=30, y=450)
#
# cgraph = tk.Button(frame1, text="Common Graph", command=window.destroy  ,fg="black"  ,bg="red"  ,width=15 ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
# cgraph.place(x=250, y=450)

quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="black", bg="red", width=35, height=1,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=120, y=500)

##################### END ######################################

window.configure(menu=menubar)
window.mainloop()

####################################################################################################


import webbrowser

# def generate_plot():
#     # Create a sample Plotly figure (you can customize this based on your data)
#     fig = make_subplots(rows=1, cols=1, subplot_titles=['Sample Plot'])
#     trace = go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='lines', name='Sample Data')
#     fig.add_trace(trace)
#
#     # Save the Plotly figure to an HTML file
#     pyo.plot(fig, filename='plot.html', auto_open=False)
#
#     # Open the saved HTML file in a web browser
#     webbrowser.open('plot.html')
#
# # Create the Tkinter window
# window = tk.Tk()
# window.title("Tkinter with Plotly")
#
# # Create a button to generate the plot
# button = ttk.Button(window, text="Generate Plot", command=generate_plot)
# button.pack(pady=20)
#
# # Start the Tkinter event loop
# window.mainloop()


# ####################################################################################################
