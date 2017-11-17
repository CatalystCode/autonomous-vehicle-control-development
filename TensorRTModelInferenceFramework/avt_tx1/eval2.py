import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
from math import exp
import httplib
import urllib
import requests
from ctypes import *
import binascii
import sys
import cv2
import socket
import cStringIO

from flask import Flask, flash, redirect, render_template, request, session, abort
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
app = Flask(__name__)

app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

lib = cdll.LoadLibrary('/home/bin/sample_cars.so')

class Classifier(object):
    def __init__(self):
        self.obj = lib.Classifier_new()

    def getclass(self):
        class_str = lib.Classifier_getclass(self.obj)
        return class_str

    def getprob(self):
        prob = lib.Classifier_getprob(self.obj)
        return prob

    def giveimg(self,img_file):
        lib.Classifier_setimg(self.obj,img_file)

    def classify(self):
        lib.Classifier_classify(self.obj)


@app.route("/")
def index():
    print "apprunning"
    img_source = "http://whitesquareproperties.com.au/wp-content/uploads/2016/08/logo.png"

    return render_template('test.html', img=img_source, pred="")

@app.route("/classify", methods=['GET','POST'])
def classify():
    img_loc = request.form['image']
    #call C++
    c = Classifier()

   
    #read in image from image location
    print "reading image from " + str(img_loc)
    im = cv2.imread(str(img_loc))

    #convert image to Float32 Array
    print "converting to float type"
    input_img = im.astype(np.float32)

    #writing image data to text file
    f = open('inputfile.txt', 'w')
    f.write(str(input_img))

    #sending image information to C++ as txt file
    c.giveimg('inputfile.txt')

    #Calling the C++ classify method
    c.classify()
    pred_str ="Prediction: "
    class_str = c.getclass()
    if class_str:
        pred_str += "No Car"
    else:
        pred_str += "Car"

    pred_str += str(c.getprob())

    return render_template('test.html', img=img_loc, pred=str(pred_str))

@app.route("/getimage", methods=['GET','POST'])
def getimage():
    #get image from web server
    c = Classifier()

    img_loc = 'http://localhost:8080'
    resp = urllib.urlopen(img_loc)
    image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)

    height, width = image.shape[:2]
    resized = cv2.resize(image, (width/8, height/8), interpolation = cv2.INTER_AREA)

    #print sys.getsizeof(resized)

    input_img = resized.astype(np.float32)
    img_floatarr = input_img/256

    #writing image data to text file
    f = open('/home/liz/upload/ifile.txt', 'w')
    #print f
    for i in img_floatarr:
        for j in i:
            for k in j:
                f.write(str(k) + "\n")

    #sending image information to C++ as txt file
    c.giveimg('ifile.txt')

    #Calling the C++ classify method
    c.classify()
    pred_str ="Prediction: "
    c.getclass()
    c.getprob()
   
    ofclass = open('outputclass.txt', 'r')
    class_str = ofclass.readline()
    ofprob = open('outputprob.txt', 'r')
    prob_str = ofprob.readline()
 
    pred_str += class_str
    pred_str += " " +  prob_str

    return render_template('test.html', img=img_loc, pred=pred_str)

if (__name__) == "__main__":
        app.run()
