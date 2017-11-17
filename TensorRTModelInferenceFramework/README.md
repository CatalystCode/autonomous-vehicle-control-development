# Python Web App for Evaluating TensorFlow Model on TensorRT

1. MakePrediction.cpp - Code for Input Evaluation
2. PredictionWebApp.py - Web application that receives input and reveals output to UI
3. WebServer-Python3.py - Web application that hosts input for evaluation

#Edits:
 - In PredictionWebApp.py, line 25: insert path to a .so file that matches your naming convention.
 - In Makefile in the same directory level as PredictionWebApp.py, lines 1 & 2, type the name of your .so file
     - This complies the C++ code into a library that the Python application has access to
 - In Makefile in the directory level above PredictionWebApp.py, line 29, type the path to your application

#Prerequisites:
 - Make sure Python 2.7 and Python 3 are installed (2.7 PredictionWebApp.py and 3 for WebServer-Python3.py)
 - Install Python dev tools:
    sudo apt-get install libfreetype6 libfreetype6-dev zlib1g-dev
 - Install Python Imaging Library:
    sudo apt-get build-dep python-imaging
    sudo apt-get install libjpeg62 libjpeg62-dev
 - Install Flask: http://flask.pocoo.org/docs/0.12/installation/
 - If you are not using the Unity application, save the image to be evaluated in your /tmp directory as Snapshot.png
