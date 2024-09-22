# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:04:45 2021

@author: MBI
"""
# Probando los metodos de la API

import requests,cv2
import numpy as np 
import matplotlib.pyplot as plt 
#%%
# Usando el metodo GET

FACE_DETECTION_REST_API_URL = "http://localhost:5000/detect"
FACE_DETECTION_REST_API_URL_WRONG = "http://localhost:5000/process"
IMAGE_PATH = "C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/face_test.jpg"

URL_IMAGE = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"

r = requests.get(FACE_DETECTION_REST_API_URL_WRONG)

print("status code: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))
#%%
# Usnado el correcto GET

payload = {'url':URL_IMAGE}
r = requests.get(FACE_DETECTION_REST_API_URL,params=payload)

print("status code: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))
#%%
# Usando el metodo GET , olvidando el payload

r = requests.get(FACE_DETECTION_REST_API_URL)

print("status code: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))
#%%
# Usando el metodo POST

image = open(IMAGE_PATH,"rb").read()
payload = {"image":image}

r = requests.post(FACE_DETECTION_REST_API_URL,files = payload)

print("status code: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))
#%%
# Usando el metodo PUT

r = requests.put(FACE_DETECTION_REST_API_URL,files = payload)

print("status code: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))
#%%
def show_matplot(image,title,pos):
    
    img_rgb = image[:,:,::-1]
    plt.subplot(1,1,pos)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')

FACE_DETECTION_REST_API_URL = "http://localhost:5000/detect"
IMAGE_PATH = "C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/face_test.jpg"

image = open(IMAGE_PATH,"rb").read()
payload = {"image":image}

r = requests.post(FACE_DETECTION_REST_API_URL,files=payload)

print("status code: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))

json_data = r.json()
result = json_data['result']

image_array = np.asarray(bytearray(image),dtype="uint8")
img_opencv = cv2.imdecode(image_array,-1)

for face in result:
    left,top,right,bottom = face['box']
    cv2.rectangle(img_opencv,(left,top),(right,bottom),(0,255,255),2)
    cv2.circle(img_opencv,(left,top),5,(0,0,255),-1)
    cv2.circle(img_opencv,(right,bottom),5,(255,0,0),-1)

fig = plt.figure(figsize=(10,5))
plt.suptitle("Usando la API de deteccion de rostros",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')
show_matplot(img_opencv,"Deteccion de rostros",1)
plt.show()

#%%
