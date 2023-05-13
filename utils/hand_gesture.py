# hand_gesture.py
import cv2
import csv
import mediapipe as mp
import numpy as np
from keras.models import load_model
import warnings

model = load_model('hand_gesture_model.h5')

# Read the labels 
labels = []

with open('Data/labels.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        labels.append(row)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def generate_video_feed():
    warnings.filterwarnings("ignore")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
            results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.append(landmark.x)
                    coords.append(landmark.y)

                coords = np.array(coords).reshape(1, -1) / 255.0
                prediction = model.predict(coords)

                label = np.argmax(prediction)
                text = str(labels[label]).strip("[]''")
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
