#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:53:17 2021

@author: mbi
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
#%% 
# Deteccion con dnn

def show_img_matplolib(img,title,pos):
    img_rgb = img[:,:,::-1]
    plt.subplot(1,1,pos)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')


network = cv2.dnn.readNetFromCaffe('C:/Users/MBI/Documents/Python Scripts/Mastering_OpenCV4/Seccion_3/deploy.prototxt','C:/Users/MBI/Documents/Python Scripts/Mastering_OpenCV4/Seccion_3/res10_300x300_ssd_iter_140000_fp16.caffemodel')

image = cv2.imread('C:/Users/MBI/Documents/Python Scripts/Mastering_OpenCV4/Seccion_3/test_face_detection.jpg')

image_c = image.copy()
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image,1.0,(300,300),[104.,117.,123.],False,False)

network.setInput(blob)
detections = network.forward()
detected_faces = 0

for i in range(0,detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > 0.7:
        detected_faces += 1
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (starX,startY,endX,endY) = box.astype('int')
        
        text = "{:.3f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image_c, (starX,startY), (endX,endY), (255,0,0),2)
        cv2.putText(image_c, text, (starX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2)

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('gray')
plt.suptitle("Deteccion completa con cv2.dnn",fontsize=14,fontweight='bold')

show_img_matplolib(image_c,"Detection with coffe",1)
plt.show()

        
#%%
# Utilizando tensorflow

netflow = cv2.dnn.readNetFromTensorflow('C:/Users/MBI/Documents/Python Scripts/Mastering_OpenCV4/Seccion_3/opencv_face_detector_uint8.pb','C:/Users/MBI/Documents/Python Scripts/Mastering_OpenCV4/Seccion_3/opencv_face_detector.pbtxt')

image_cc = image.copy()
blob_flow = cv2.dnn.blobFromImage(image,1.0,(300,300),[104.,117.,123.],False,False)

netflow.setInput(blob_flow)
detections_flow = netflow.forward()
detected_face = 0

for i in range(0,detections_flow.shape[2]):
    confidence = detections_flow[0,0,i,2]
    if confidence > 0.7:
        detected_face += 1
        box = detections_flow[0,0,i,3:7] * np.array([w,h,w,h])
        (starX,startY,endX,endY) = box.astype('int')
        
        text = "{:.3f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image_cc, (starX,startY), (endX,endY), (255,0,0),2)
        cv2.putText(image_cc, text, (starX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2)

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('gray')
plt.suptitle("Deteccion completa con cv2.dnn",fontsize=14,fontweight='bold')

show_img_matplolib(image_c,"Detection with tensorflow",1)
plt.show()


#%%
# Deteccion usando cropp

def show_img_matplolib(img,title,pos):
    img_rgb = img[:,:,::-1]
    plt.subplot(2,2,pos)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')

def get_cropped_image(image):
    
    imgs_cropped = []
    for img in image:
        img_copy = img.copy()
        size = min(img_copy.shape[1],img_copy.shape[0])
        x1 = int(0.5 * (img_copy.shape[1] - size))
        y1 = int(0.5 * (img_copy.shape[0] - size))
        imgs_cropped.append(img_copy[y1:(y1 + size),x1:(x1 + size)])
    return imgs_cropped


image = cv2.imread('C:/Users/MBI/Documents/Python Scripts/Mastering_OpenCV4/Seccion_3/face_test.jpg')
image_ = cv2.imread('C:/Users/MBI/Documents/Python Scripts/Mastering_OpenCV4/Seccion_3/face_test_2.jpg')
images = [image,image_]

images_cropped = get_cropped_image(images)

blob_cropped = cv2.dnn.blobFromImages(images,1.0,(300,300),[104.,117.,123.],False,True)
print(blob_cropped.shape)
net_cropped = cv2.dnn.readNetFromCaffe('C:/Users/MBI/Documents/Python Scripts/Mastering_OpenCV4/Seccion_3/deploy.prototxt','C:/Users/MBI/Documents/Python Scripts/Mastering_OpenCV4/Seccion_3/res10_300x300_ssd_iter_140000_fp16.caffemodel')

net_cropped.setInput(blob_cropped)
detections_cropped = net_cropped.forward()

for i in range(0,detections_cropped.shape[2]):
    img_id = int(detections_cropped[0,0,i,0])
    confidence = detections_cropped[0,0,i,2]
    if confidence > 0.25:
        (h,w) = images_cropped[img_id].shape[:2]
        box = detections_cropped[0,0,i,3:7] * np.array([w,h,w,h])
        (starX,startY,endX,endY) = box.astype('int')
        
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(images_cropped[img_id], (starX,startY), (endX,endY), (255,0,0),2)
        cv2.putText(images_cropped[img_id], text, (starX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2)
        

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('brown')
plt.suptitle("Deteccion completa con cv2.dnn using cropped images",fontsize=14,fontweight='bold')

show_img_matplolib(images[0],"Image original",1)
show_img_matplolib(images[1],"Image original",2)
show_img_matplolib(images_cropped[0],"Image cropped",3)
show_img_matplolib(images_cropped[1],"Image original",4)
plt.show()

#%%

