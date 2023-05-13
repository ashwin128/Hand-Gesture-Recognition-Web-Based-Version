import mediapipe as mp
import cv2
import os
import csv

def extract_landmarks(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if not results.multi_hand_landmarks:
            return None
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y))
        return landmarks

def main():
    csv_file = 'hand_landmarks.csv'
    with open(csv_file, mode='w', newline='') as f:
        number_of_classes = len(os.listdir('Data')) - 1
        writer = csv.writer(f)
        for i in range(number_of_classes):
            print("Training Hand Gesture", i)
            subfolder = os.path.join('Data', str(i))
            for j in range(300):
                image_path = os.path.join(subfolder, f'{j}.jpg')
                image = cv2.imread(image_path)
                landmarks = extract_landmarks(image)
                if landmarks:
                    writer.writerow([i] + [coord for landmark in landmarks for coord in landmark])
