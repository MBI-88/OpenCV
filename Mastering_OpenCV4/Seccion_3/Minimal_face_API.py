# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 20:01:54 2021

@author: MBI
"""

# Minimal Face API using OpenCV
import urllib.request 
from  flask import Flask,request,jsonify
import cv2,os
import numpy as np
#%%
# Definicion de la clase FaceProcessing

class FaceProcessing(object):
    def __init__(self):
        self.file = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
# Opcion del libro: s.path.join(os.path.join(os.path.dirname(__file__),"data"), "haarcascade_frontalface_alt.xml")
        self.face_cascade = cv2.CascadeClassifier(self.file)
    
    def face_detector(self,image):
        image_array = np.asarray(bytearray(image),dtype=np.uint8)
        img_opencv = cv2.imdecode(image_array, -1)
        output = []
        gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGRA2GRAY)
        faces = self.face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(25,25))
        
        for face in faces:
            x,y,w,h = face.tolist()
            face = {"box":[x,y,x+w,y+h]}
            output.append(face)
        return output

app = Flask(__name__)
fc = FaceProcessing()

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status":"not ok","message":"this server could not understand your request"}),400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status":"not found","message":"route not found"}),404

@app.errorhandler(500)
def not_found(e):
    return jsonify({"status":"internal error","message":"internal  error  occured in server"}),500

@app.route("/detect",methods=['GET','POST','PUT'])
def detect_human_face():
    if request.method == 'GET':
        if request.args.get('url'):
            with urllib.request.urlopen(request.args.get('url')) as url:
                return jsonify({"status":"ok","result":fc.face_detector(url.read())}),200
        else:
            return jsonify({"status":"bad request","message":"Parameter url is not present"}),400
    
    elif request.method == 'POST':
        if request.files.get('image'):
            return jsonify({"status":"ok","result":fc.face_detector(request.files['image'].read())}),200
        else:
            return jsonify({"status":"bad request","message":"Parameter image is not  present"}),400
    else:
        return jsonify({"status":"failure","message":"PUT method not supported for API"}),405


if __name__ == '__main__':
    app.run(host='0.0.0.0')        
        
        
#%%

