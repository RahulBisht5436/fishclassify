from __future__ import division, print_function

from flask import Flask, render_template, url_for, redirect,request
import sqlite3

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import PIL.Image 

import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import io

model=tf.keras.models.load_model('rahulbishtfinalproject.h5')

app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def home():
    if request.method=="POST":
        

        name=request.form['name']
        password=request.form['password']

        connection=sqlite3.connect('user_data.db')
        cursor=connection.cursor()
        query="SELECT name,password FROM user where name='"+name+"' and password='"+password+"'"
        cursor.execute(query)
        result =cursor.fetchall()



        if len(result)==0:
            print('wrong password')
            return render_template('home.html',name='incorrect password or namae')
        else:
            return redirect('pred')
        
    return render_template('home.html',)

@app.route('/reg',methods=["GET","POST"])
def reg():
     if  request.method =='POST':
            if request.form['name']!="" and request.form['password']!="":
                 name=request.form['name']
                 password=request.form['password']
                 connection=sqlite3.connect('user_data.db')
                 cursor=connection.cursor()
                 query= "INSERT INTO user VALUES('"+name+"','"+password+"')"
                 cursor.execute(query)
                 connection.commit()
                 return redirect('/')
            
     return render_template('registration.html')
               

            

def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size "size"
    array = np.expand_dims(array, axis=0)
    return array

def model_predict(img_path, model):
    #preprocessing
    test=get_img_array(img_path,(256,256,3))
    pred=model.predict(test)
    pred=np.argmax(pred,axis=1)
    return pred


@app.route('/pred', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
globalvar=0

@app.route('/pred', methods=[ 'POST'])
def upload():
    if request.method == 'POST':
        # # Get the file from post request
        # f = request.files['file']
        # image=f.data()
        # # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)

        # Make prediction
        # preds = model_predict(file_path, model)
        file = request.files['file']

    # read the file data into memory
        file_bytes = io.BytesIO(file.read())
        img = tf.keras.utils.load_img(file_bytes, target_size=(256, 256))
    # open the image file with PIL
        # img = PIL.Image.open(file_bytes)

    # resize the image
        # size = (256, 256)  # set the new size of the image
        # img = img.resize(size)
        
        # img = cv2.resize(img, (256,256))
        img = np.array(img)
        img = np.reshape(img, (1,256,256,3))
        pred=model.predict(img)
        preds=np.argmax(pred,axis=1)

        print(f'ama********************************************${preds}**************************************************')
        if  int(int(preds)==0):
            return render_template('infoblackseasprat.html')
        if int(int(preds)==1):
            return render_template('infoglitredfish.html')
        if int(int(preds)==2):
            return render_template('horsemac.html')
        if int(int(preds)==3):
            return render_template('inforedmullet.html')
        if int(int(preds)==4):
            return render_template('inforedseabeam.html')
        if int(int(preds)==5):
            return render_template('infoseabass.html')
        if int(int(preds)==6):
            return render_template('infoshrimp.html')
        if int(int(preds)==7):
            return render_template('infostrippedredmullet.html')
        if int(int(preds)==8):
            return render_template('infotrout.html')
        
          
    return None

@app.route('/d0', methods=['GET'])
def d0():
    return render_template('d1.html')

@app.route('/d1', methods=['GET'])
def d1():
    return render_template('d0.html')

@app.route('/d2', methods=['GET'])
def d2():
    return render_template('d2.html')

@app.route('/d3', methods=['GET'])
def d3():
    return render_template('d3.html')

@app.route('/d4', methods=['GET'])
def d4():
    return render_template('d4.html')

@app.route('/d5', methods=['GET'])
def d5():
    return render_template('d5.html')

@app.route('/d6', methods=['GET'])
def d6():
    return render_template('d6.html')

@app.route('/d7', methods=['GET'])
def d7():
    return render_template('d7.html')

@app.route('/d8', methods=['GET'])
def d8():
    return render_template('d8.html')

if __name__ == '__main__':
    app.run(debug=True)



            
         
     
        
    

if __name__ == "__main__":
    app.run(debug=True)
