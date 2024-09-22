# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:42:48 2021

@author: MBI
"""
from flask import Flask,request,jsonify
import urllib.request,cv2
import numpy as np
#%%
class ImageProcessin(object):
    def __init__(self):
        self.file = cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml"
        self.cat_cascade = cv2.CascadeClassifier(self.file)
        self.net_cat = cv2.dnn.readNetFromCaffe("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/MobileNetSSD_deploy.prototxt","C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/MobileNetSSD_deploy.caffemodel")
             
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                        "car", "cat", "chair", "cow", "diningtable", "dog", "horse","motorbike",
                        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    
    def cat_face_detection(self,image):
        image_array = np.asarray(bytearray(image),dtype='uint8')
        img_opencv = cv2.imdecode(image_array,-1)
        output = []
        gray = cv2.cvtColor(img_opencv,cv2.COLOR_BGR2GRAY)
        cats = self.cat_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(25,25))
        for cat in cats:
            x,y,w,h = cat.tolist()
            face = {"box":[x,y,x+w,y+h]}
            output.append(face)
        return output
    
    def cat_dection(self,image):
        image_array = np.asarray(bytearray(image),dtype='uint8')
        img_opencv = cv2.imdecode(image_array,-1)
        blob = cv2.dnn.blobFromImage(img_opencv,0.007843,(300,300),(127.5,127.5,127.5))
        
        self.net_cat.setInput(blob)
        detections = self.net_cat.forward()
        dim = 300
        output = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > 0.1 :
                class_id = int(detections[0,0,i,1])
                left = int(detections[0,0,i,3] * dim)
                top = int(detections[0,0,i,4] * dim)
                right = int(detections[0,0,i,5] * dim)
                bottom = int(detections[0,0,i,6] * dim)
                
                heightFactor = img_opencv.shape[0] / dim
                weightsFactor = img_opencv.shape[1] / dim 
                
                left = int(weightsFactor * left)
                top = int(heightFactor * top)
                right = int(weightsFactor * right)
                bottom = int(heightFactor  * bottom)
                
                if self.classes[class_id] == 'cat':
                    cat = {"box":[left,top,right,bottom]}
                    output.append(cat)
        return output

app = Flask(__name__)
ip = ImageProcessin()

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status":"not ok","message":"this server could not understand your request"}),400 

@app.errorhandler(404)
def not_foud(e):
    return jsonify({"status":"not found","message":"route  not found"}),404 

@app.errorhandler(500)
def not_found(e):
    return jsonify({"status":"internal error","message":"internal error ocurred in server"}),500 

@app.route("/catfacedetection",methods=['GET','POST','PUT'])
def detect_cat_face():
    if request.method == 'GET':
        if request.args.get('url'):
            with urllib.request.urlopen(request.args.get('url'))  as url:
                return jsonify({"status":"ok","result":ip.cat_face_detection(url.read())}),200 
        else:
            return jsonify({"status":"bad request","message":"Parameter url is not present"}),400 
    
    elif request.method == 'POST':
        if request.files.get('image'):
            return jsonify({"status":"ok","result":ip.cat_face_detection(request.files["image"].read())}),200 
        
        else:
            return jsonify({"status":"bad request","message":"Parameter image is not present"}),400
        
    else:
        return jsonify({"status":"failure","message":"PUT method not supported for API"}),405 


@app.route("/catdetection",methods=['GET','POST','PUT'])
def detect_cat():
    if request.method == 'GET':
        if request.args.get('url'):
            with urllib.request.urlopen(request.args.get('url')) as url:
                return jsonify({"status":"ok","result":ip.cat_dection(url.read())}),200 
        else:
            return jsonify({"status":"bad request","message":"Parameter url is not present"}),400 
    
    elif request.method == 'POST':
        if request.files.get('image'):
            return jsonify({"status":"ok","result":ip.cat_dection(request.files['image'].read())}),200 
        else :
            return jsonify({"status":"bad request","message":"Parameter image is no present"}),400 
    else:
        return jsonify({"status":"failure","message":"PUT method not supported for API"}),405 


if __name__ == '__main__':
    app.run(host='0.0.0.0' )
#%%
        
            
                
                
                
                
                


