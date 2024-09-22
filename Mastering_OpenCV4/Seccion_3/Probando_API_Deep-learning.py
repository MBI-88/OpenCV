# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:14:25 2021

@author: MBI
"""
import cv2,requests 
import numpy as np 
import matplotlib.pyplot as plt 
#%%
def show_matplotlib(image,title,pos):
    img_rgb = image[:,:,::-1]
    plt.subplot(1,1,pos)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")


CAT_FACE_DETECTION_REST_API_URL = "http://localhost:5000/catfacedetection"
CAT_DETECTION_REST_API_URL = "http://localhost:5000/catdetection"
IMG_PATH = "C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/cat-1.jpg"

image = open(IMG_PATH,"rb").read()
payload = {"image":image}

image_array = np.asarray(bytearray(image),dtype='uint8')
img_opnecv = cv2.imdecode(image_array,-1)

r = requests.post(CAT_DETECTION_REST_API_URL,files=payload)

print("stus: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))
print("\n")
json_data = r.json()
result = json_data['result']

for cat in result:
    left,top,right,bottom = cat["box"]
    cv2.rectangle(img_opnecv,(left,top),(right,bottom),(0,255,0),2)
    cv2.circle(img_opnecv,(left,top),10,(0,0,255),-1)
    cv2.circle(img_opnecv,(right,bottom),10,(0,0,255),-1)

r = requests.post(CAT_FACE_DETECTION_REST_API_URL,files=payload)

print("stus: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))

json_data = r.json()
result = json_data['result']

for face in result:
    left,top,right,bottom = face["box"]
    cv2.rectangle(img_opnecv,(left,top),(right,bottom),(0,255,0),2)
    cv2.circle(img_opnecv,(left,top),10,(0,0,255),-1)
    cv2.circle(img_opnecv,(right,bottom),10,(0,0,255),-1)

fig = plt.figure(figsize=(10,6))
plt.suptitle("Using cat detection API",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

show_matplotlib(img_opnecv,"Cat detection",1)
plt.show()



#%%