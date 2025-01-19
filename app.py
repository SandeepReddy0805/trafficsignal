
import os
import time
from flask import Flask, render_template, request
import numpy as np

from ultralytics import YOLO
import argparse
import io
import os
from PIL import Image
import datetime

import torch
from flask import Flask, render_template, request, redirect
import io
import pandas as pd
import numpy as np

import random
import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect


import supervision as sv

app = Flask(__name__)

app.config["PATH_TO_UPLOAD"] = os.path.join("static", "img")

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

model_8 = YOLO('best.pt')  # force_reload = recache latest code


@app.route('/index')
def index():
	return render_template('index.html')

@app.route("/predict", methods=["POST","GET"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
    
    
        #print('Model 8')
        
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model_8(img)
        
        res_plotted = results[0].plot()
        result = model_8(img)[0]
        detections = sv.Detections.from_ultralytics(result)
        #print(len(detections))
        dens = len(detections)
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            masks = result.masks  # Masks object for segmenation masks outputs
            probs = result.probs  # Class probabilities for classification outputs
            # updates results.imgs with boxes and labels
            #now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        for box in boxes: #there could be more than one detection
            box=box.numpy()
            #print(box[0]) 
            pred = model_8.names.get(box.cls.item())
        print(pred)
        #img_savename = f"static/image0.png"
        img_filename = str(int(time.time())) + '.jpg'
        img_base64 = Image.fromarray(res_plotted)
        img_base64.save(f'static/{img_filename}', format='JPEG')
        if pred == 'ambulance':
            output = "Green Signal! Since Ambulance Detected! "
            return render_template("result.html",output = output,img_filename=img_filename)

        else:
            if dens <= 20:
                output = "Red Signal! Density of Vehicle is less Than 20!"
                return render_template("result1.html",dens = dens,output = output, img_filename=img_filename)
            elif dens > 20:
                output = "Green Signal! Density of Vehicle is More Than 20!"
                return render_template("result1.html",dens = dens,output = output, img_filename=img_filename)
        
        #Image.fromarray(res_plotted).save(img_savename)
        return render_template("result.html", img_filename=img_filename)
    
    
    return render_template('index.html')
        
    
@app.route('/')
@app.route('/home')
def home():
	return render_template('index.html')


if __name__ == "__main__":
    
    app.run(debug=False)  # debug=True causes Restarting with stat
