import cv2
import mediapipe as mp

def detect_handraise_earphones_face():
    # Load the pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Check if faces are detected
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Assume earphones are near the ears (adjust the coordinates based on your assumptions)
            ear_region_left = frame[y:y+h, x-20:x]
            ear_region_right = frame[y:y+h, x+w:x+w+20]

            # Check if there are non-zero pixels in the ear regions
            if cv2.countNonZero(cv2.cvtColor(ear_region_left, cv2.COLOR_BGR2GRAY)) > 10 or \
               cv2.countNonZero(cv2.cvtColor(ear_region_right, cv2.COLOR_BGR2GRAY)) > 10:
                cv2.putText(frame, "Earphones Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Earphones Not Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Hand-raise detection logic (you can replace this with your hand-raise detection code)
            hand_raise_threshold = 60
            hand_region = gray[y:y+h, x:x+w]
            _, hand_threshold = cv2.threshold(hand_region, hand_raise_threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            hand_contours, _ = cv2.findContours(hand_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(hand_contours) > 0:
                cv2.putText(frame, "Hand Raised!", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Hand Raised", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Hand and Earphones Detection (Face)', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # Loop through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw green dots on each landmark
            for landmark in hand_landmarks.landmark:
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    # Display the annotated image
    cv2.imshow("Hand Landmarks", image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
