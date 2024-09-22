# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:43:01 2021

@author: MBI
"""
import cv2 
import numpy as np
from flask import Flask,request,make_response
import urllib.request
#%%

app = Flask(__name__)
@app.route('/canny',methods=['GET'])
def canny_processing():
    with urllib.request.urlopen(request.args.get('url')) as url:
        image_array = np.asarray(bytearray(url.read()),dtype=np.uint8)
    
    img_cv = cv2.imdecode(image_array, -1)
    gray = cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    retval,buffer = cv2.imencode('.jpg',edges)
    respose = make_response(buffer.tobytes())
    respose.headers['Content-Type'] = 'image/jpeg'
    return respose

if __name__ == '__main__':
    app.run(host='0.0.0.0')
# Ingresar en el navegador: http://ip de la maquina :5000/canny? url=https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg