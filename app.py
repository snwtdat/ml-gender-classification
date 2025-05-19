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
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128,128))
    img = img.reshape(1,128,128,1) / 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)

        img = preprocess_image(path)
        pred = model.predict(img)[0][0]
        gender = "Male" if pred > 0.5 else "Female"

        return render_template('index.html', filename=f.filename, gender=gender)

    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return f'/static/uploads/{filename}'

if __name__ == '__main__':
    app.run(debug=True)
