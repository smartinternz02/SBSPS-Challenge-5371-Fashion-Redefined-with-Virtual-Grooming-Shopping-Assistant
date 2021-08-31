import numpy as np
import os
import cv2
import imageio
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model("MODEL30AUG.h5")
                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        im = imageio.imread(filepath)
        im = cv2.resize(im,(28,28),3)
        gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
        img_rows, img_cols = 28, 28
        gray = gray.reshape(1, img_rows, img_cols, 1)
        gray /= 255
        prediction = model.predict(gray)
        n=prediction.argmax()
        print("prediction",n)
        index = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        text = "The Shown Apparel is : " + str(index[n])
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
