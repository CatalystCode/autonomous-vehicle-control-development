# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
# Based on code provided by NVIDIA's JetPack Sample

import numpy as np
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

lib = cdll.LoadLibrary('/path/to/shared/object/file')

class Evaluator(object):
    def __init__(self):
        self.obj = lib.Evaluator_new()

    def geteval(self):
        class_str = lib.Evaluator_geteval(self.obj)
        return class_str

    def getprob(self):
        prob = lib.Evaluator_getprob(self.obj)
        return prob

    def giveimg(self,img_file):
        lib.Evaluator_setimg(self.obj,img_file)

    def evaluate(self):
        lib.Evaluator_evaluate(self.obj)


@app.route("/")
def index():
    print "apprunning"
    img_source = "http://whitesquareproperties.com.au/wp-content/uploads/2016/08/logo.png"

    return render_template('test.html', img=img_source, pred="")

@app.route("/getimage", methods=['GET','POST'])
def getimage():
    #declare a new evaluator instance
    e = Evaluator()

    #get image from web server
    img_loc = 'http://192.168.88.250:8080'
    resp = urllib.urlopen(img_loc)
    image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)

    #resize the input, based on what the UFF file is expecting as input
    height, width = image.shape[:2]
    #img = cv2.resize(image, (width/8, height/8), interpolation = cv2.INTER_AREA)
    img = cv2.resize(image, (224,224))

    input_img = img.astype(np.float32)
    img_floatarr = input_img/256

    #writing image data to text file
    f = open('/home/nvidia/avt/avt_tx1/ifile.txt', 'w')
    for i in img_floatarr:
        for j in i:
            for k in j:
                f.write(str(k) + "\n")

    f.close()

    #sending image information to C++ as txt file
    e.giveimg('ifile.txt')

    #Calling the C++ evaluate method
    e.evaluate()
    pred_str ="Prediction: "
    e.geteval()
    e.getprob()
   
    #writing the final evaluation to output files
    ofeval = open('outputeval.txt', 'r')
    eval_str = ofeval.readline()
    ofprob = open('outputprob.txt', 'r')
    prob_str = ofprob.readline()

    ofeval.close()
    ofprob.close()
 
    pred_str += eval_str
    pred_str += " " +  prob_str

    return render_template('test.html', img=img_loc, pred=pred_str)

if (__name__) == "__main__":
        app.run(host='0.0.0.0')
