############################################# IMPORTING ################################################
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

############################################# FUNCTIONS ################################################

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
    mess._show(title='Contact us', message="Please contact us on : 'xxxxxxxxxxxxx@gmail.com' ")


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


def clear2():
    txt2.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)


path = 'TrainingImage'
image = []
idNames = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    image.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    idNames.append(os.path.splitext(cl)[1])
print(classNames)

#######################################################################################

# using the haar cascade for detecting the face and then doing the singUP
def TakeImages():
    print("In function take image")
    check_haarcascadefile()
    # To save the file in this particular format.
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    # if the path for studentDetails exists then read the current file and get serial else create and set serial as 1
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
        # Open Cam... 0 means webCam.
        cam = cv2.VideoCapture(0)
        # We are using haar-cascade pre trained classifier to detect the faces in a particular frame
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            # Getting a frame using read(), It returns a tuple of status, and a image. so check the status then do stuff on image
            ret, img = cam.read()
            # Converting the above read image to gray.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # using function of cascade classifier we
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ " + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('Taking Images', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 100:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Taken for ID : " + Id
        row = [serial, '', Id, '', name]
        # Updating the result in the studentDetails csv file.
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        # Printing message on GUI
        message1.configure(text=res)
    else:
        if (name.isalpha() == False):
            res = "Enter Correct name"
            message.configure(text=res)


########################################################################################
def find_encodings(images):
    encode_list = []
    cnt = 0
    for img_1 in images:
        try:
            cnt+=1
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img_1)[0]
            encode_list.append(encode)
        except:
            print("Processing...")
    return encode_list

# face detection object is returned by get_frontal_face_detector() function of DLIB library.
face_detector = dlib.get_frontal_face_detector()

# shape_predictor_68_face_landmarks.dat is a pre-Trained Model file
# this has annotation of 68 Facial landmarks
# now landmark_predictor can be used to detect the face (img, face) here, img is the image and face is the region of face
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

activity_history = deque(maxlen=100)  # Store activity history for plotting
head_down_threshold = 3  # Threshold for head down
eye_aspect_ratio_threshold = 0.2  # Threshold for eye closed

# File to store time-to-time scores
score_file = open("student_scores.txt", "w")
scoring_interval = 60  # e.g., score every 60 seconds
timestamps = []
scores = []

def angle_between_points(p1, p2, p3):
    angle = abs(p2 - p1) - abs(p2 - p3)
    return angle


# When we click on "take Attendance" then it is called
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


############################################################################################3

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


###########################################################################################

def TrackImages():
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
            # print(f"#######{image[best_match_index]} ##########")


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
                mess._show(title='Details Missing', message='Students details are missing, please check!')

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

                attentiveness_data = attentiveness_data.append({'Timestamp': time.time(), 'Attentiveness': score}, ignore_index=True)

                # Reset variables for the next scoring interval
                activity_history.clear()
                start_time = time.time()
        except:
            print("Adding in file...")
        cv2.putText(frame, f"Head Status: {head_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
                    tv.insert('', 0, text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
    csvFile1.close()


    print("In function Track Images")


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
window.geometry("1280x720")
window.resizable(True, False)
window.title("Attendance System")
window.configure(background='#262523')

frame1 = tk.Frame(window, bg="#00aeff")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#00aeff")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="Face Recognition Based Attendance System", fg="white", bg="#262523", width=55,
                    height=1, font=('times', 29, ' bold '))
message3.place(x=10, y=10)

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4, text=day + "-" + mont[month] + "-" + year + "  |  ", fg="orange", bg="#262523", width=55,
                 height=1, font=('times', 22, ' bold '))
datef.pack(fill='both', expand=1)

clock = tk.Label(frame3, fg="orange", bg="#262523", width=55, height=1, font=('times', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

head2 = tk.Label(frame2, text="                       For New Registrations                       ", fg="black",
                 bg="#3ece48", font=('times', 17, ' bold '))
head2.grid(row=0, column=0)

head1 = tk.Label(frame1, text="                       For Already Registered                       ", fg="black",
                 bg="#3ece48", font=('times', 17, ' bold '))
head1.place(x=0, y=0)

lbl = tk.Label(frame2, text="Enter ID", width=20, height=1, fg="black", bg="#00aeff", font=('times', 17, ' bold '))
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", font=('times', 15, ' bold '))
txt.place(x=30, y=88)

lbl2 = tk.Label(frame2, text="Enter Name", width=20, fg="black", bg="#00aeff", font=('times', 17, ' bold '))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", font=('times', 15, ' bold '))
txt2.place(x=30, y=173)

message1 = tk.Label(frame2, text="1)Take Images  >>>  2)Save Profile", bg="#00aeff", fg="black", width=39, height=1,
                    activebackground="yellow", font=('times', 15, ' bold '))
message1.place(x=7, y=230)

message = tk.Label(frame2, text="", bg="#00aeff", fg="black", width=39, height=1, activebackground="yellow",
                   font=('times', 16, ' bold '))
message.place(x=7, y=450)

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
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading('#0', text='ID')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

###################### SCROLLBAR ################################

scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

###################### BUTTONS ##################################

clearButton = tk.Button(frame2, text="Clear", command=clear, fg="black", bg="#ea2a2a", width=11,
                        activebackground="white", font=('times', 11, ' bold '))
clearButton.place(x=335, y=86)
clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="black", bg="#ea2a2a", width=11,
                         activebackground="white", font=('times', 11, ' bold '))
clearButton2.place(x=335, y=172)
takeImg = tk.Button(frame2, text="Take Images", command=TakeImages, fg="white", bg="blue", width=34, height=1,
                    activebackground="white", font=('times', 15, ' bold '))
takeImg.place(x=30, y=300)
trainImg = tk.Button(frame2, text="Save Profile", command=psw, fg="white", bg="blue", width=34, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trainImg.place(x=30, y=380)
trackImg = tk.Button(frame1, text="Take Attendance", command=TrackImages, fg="black", bg="yellow", width=35, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trackImg.place(x=30, y=50)

# igraph = tk.Button(frame1, text="Individual Graph", command=window.destroy  ,fg="black"  ,bg="red"  ,width=15 ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
# igraph.place(x=30, y=450)
#
# cgraph = tk.Button(frame1, text="Common Graph", command=window.destroy  ,fg="black"  ,bg="red"  ,width=15 ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
# cgraph.place(x=250, y=450)

quitWindow = tk.Button(frame1, text="Quit", command=window.destroy  ,fg="black"  ,bg="red"  ,width=35 ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
quitWindow.place(x=30, y=500)

##################### END ######################################

window.configure(menu=menubar)
window.mainloop()

####################################################################################################



import webbrowser

def generate_plot():
    # Create a sample Plotly figure (you can customize this based on your data)
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Sample Plot'])
    trace = go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='lines', name='Sample Data')
    fig.add_trace(trace)

    # Save the Plotly figure to an HTML file
    pyo.plot(fig, filename='plot.html', auto_open=False)

    # Open the saved HTML file in a web browser
    webbrowser.open('plot.html')

# Create the Tkinter window
window = tk.Tk()
window.title("Tkinter with Plotly")

# Create a button to generate the plot
button = ttk.Button(window, text="Generate Plot", command=generate_plot)
button.pack(pady=20)

# Start the Tkinter event loop
window.mainloop()


# ####################################################################################################