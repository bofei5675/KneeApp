from __future__ import division, print_function
# coding=utf-8
import os
import time
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import torch
import torch.nn as nn
import numpy as np
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/your_model.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
from src.models import ResidualNet, ResNet18
from src.preprocessing import image_preprocessing, detect_preprocessing, padding
from src.utils import drawFigure, getKneeWithBbox, model_predict

print('Model loaded. Check http://127.0.0.1:5000/')
print('Load detector ...')
cur = time.time()
detector = ResNet18(pretrained=True, dropout=0.4, use_cuda=False)
detector.load_state_dict(torch.load('./models/detection/epoch_45.pth', map_location='cpu'))
detector.eval()
print('Finish load detector:', time.time() - cur)
cur = time.time()
print('Load classifier ...')
model = ResidualNet('ImageNet',34,1000,'CBAM')
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 5))
model.load_state_dict(torch.load('./models/clf/epoch_9.pth', map_location='cpu'))
model.eval()
print('Finish classifier:', time.time() - cur)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        img_path = os.path.join(basepath,'uploads', f.filename + '_img.png')
        f.save(file_path)
        # Make prediction
        print('Preprocessing')
        cur = time.time()
        img, img_copy, row, col, ratio_x, ratio_y = detect_preprocessing(file_path)
        print('Preprocessing takes {}'.format(time.time() - cur))
        cur = time.time()
        output = detector(img)
        output = output.squeeze(0).detach().numpy()
        print(output)
        drawFigure(img_copy, output, img_path)
        print('Draw figures takes {}'.format(time.time() - cur))
        cur = time.time()
        print('Detection Finished!')
        img2crop, data, img_before = image_preprocessing(file_path)
        left, right = getKneeWithBbox(img2crop, output)
        left, _, _ = padding(left, img_size=(1024, 1024))
        right, _, _ = padding(right, img_size=(1024, 1024))
        pred_left, pred_right = model_predict(model, left, right)
        print('Prediction takes {}'.format(time.time() - cur))
        return str(output) + str(pred_left) + str(pred_right)
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    # Serve the app with gevent
    http_server = WSGIServer(('', 5001), app)
    http_server.serve_forever()
