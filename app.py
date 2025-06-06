from flask import Flask, request, render_template
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("cnn_model.h5")
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(path):
    # Load ảnh màu
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        return None

    # Chọn khuôn mặt lớn nhất
    (x, y, w, h) = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (128, 128))
    face_normalized = face_resized / 255.0

    return face_normalized.reshape(1, 128, 128, 1)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)

        img = preprocess_image(path)
        pred = model.predict(img)[0][0]
        gender = "Female" if pred > 0.5 else "Male"

        return render_template('index.html', filename=f.filename, gender=gender)

    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return f'/static/uploads/{filename}'

if __name__ == '__main__':
    app.run(debug=True)
