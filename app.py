import io
import os
import csv
import uuid
import subprocess
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, Response
from utils.hand_gesture import generate_video_feed
import utils.Train as Train_Model
import utils.Create_Dataset as Create_Dataset

app = Flask(__name__)

# Read the labels 
labels = []

with open('Data/labels.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        labels.append(row)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    Create_Dataset.main()
    Train_Model.main()
    return 'Model Trained.', 200

@app.route('/hand_gesture')
def hand_gesture():
    return render_template('hand_gesture.html', labels=labels)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_images')
def take_images():
    return render_template('take_images.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    class_num = request.form['class_num']
    name = request.form['name']
    DATA_DIR = './Data'
    class_dir = os.path.join(DATA_DIR, class_num)

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Write name to labels.csv file
    csv_path = os.path.join(DATA_DIR, 'labels.csv')
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                rows.append(row)
    if len(rows) <= int(class_num):
        for i in range(len(rows), int(class_num)+1):
            rows.append([''])
    rows[int(class_num)][0] = name
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)

    images = request.files.getlist('file')
    for i, image in enumerate(images):
        image_data = image.read() # read the blob data
        image_stream = io.BytesIO(image_data) # convert to an io stream
        image_pil = Image.open(image_stream) # convert to a PIL image object
        image_pil = image_pil.convert('RGB') # convert to RGB mode
        file_name = secure_filename(f"{i}.jpg")
        file_path = os.path.join(class_dir, file_name)
        image_pil.save(file_path)

    return 'Images saved.', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

