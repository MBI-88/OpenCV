# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:58:31 2021

@author: MBI
"""
import cv2
import numpy as np
#%% Definicion del modelo

model = cv2.dnn.readNetFromCaffe("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/MobileNetSSD_deploy.prototxt","C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/MobileNetSSD_deploy.caffemodel")

# Procesamiento necesario para usar este modelo
blob_hight = 300 # lo que pide de alto 
color_scale = 1.0/127.5 # scalado necesario  entre -+1
average_color = (127.5,127.5,127.5)
confidence_threshold = 0.5

# el modelo soporta 20 clases
labels = ['airplane','bicycle','bird','boat','bottle','bus','car', 'cat', 'chair','cow','dining table','dog','horse','motorbike', 'person', 'potted plant', 'sheep','sofa', 'train', 'TV or monitor']

# computando los frame de la camara
cap = cv2.VideoCapture(0)
succes,frame = cap.read()
while succes:
    h,w = frame.shape[:2]
    aspect_ratio = w/h
    blob_width = int(blob_hight * aspect_ratio)
    blob_size = (blob_width,blob_hight)
    
    blob = cv2.dnn.blobFromImage(frame,scalefactor=color_scale,size=blob_size,mean=average_color)
    model.setInput(blob)
    results  = model.forward()
    
    for object in results[0,0]:
        confidence = object[2]
        if confidence > confidence_threshold :
            x0,y0,x1,y1 = (object[3:7] * [w,h,w,h]).astype(int)
            id = int(object[1])
            label = labels[id - 1]
            
            cv2.rectangle(frame, (x0,y0), (x1,y1), (255,0,255),2)
            text = " {}-{}".format(label,confidence * 100.0)
            cv2.putText(frame, text, (x0,y0-20), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            
    cv2.imshow("Objetc detection", frame)
    if (cv2.waitKey(1) == ord('q')):
        break
    succes,frame = cap.read()

cv2.destroyAllWindows()
#%% Detectar y clasificar rostros

face_model = cv2.dnn.readNetFromCaffe("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/deploy.prototxt","C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/res10_300x300_ssd_iter_140000.caffemodel")

face_blob_height = 300 
face_average_color = (104,117,123)
face_confidence_threshold = 0.995

age_model = cv2.dnn.readNetFromCaffe("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/age_net_deploy.prototxt","C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/age_net.caffemodel")

age_labels = ['0-2','4-6','8-12','15-20','25-32', '38-43','48-53','60+']

gender_model = cv2.dnn.readNetFromCaffe("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/gender_net_deploy.prototxt","C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/gender_net.caffemodel")

gender_labels = ['male','female']
age_gender_blob_size = (256,256)
age_gender_average_image = np.load("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/average_face.npy")

cap = cv2.VideoCapture(0)
succes,frame = cap.read()
while succes:
    h,w = frame.shape[:2]
    aspect_ratio = w/h
    face_blob_width = int(face_blob_height * aspect_ratio)
    face_blob_size = (face_blob_width,face_blob_height)
    
    face_blob = cv2.dnn.blobFromImage(frame,size=face_blob_size,mean=face_average_color)
    face_model.setInput(face_blob)
    face_results = face_model.forward()
    
    for face in face_results[0,0]:
        face_confidence = face[2]
        if face_confidence > face_confidence_threshold:
            x0,y0,x1,y1 = (face[3:7] * [w,h,w,h]).astype(int)
            y1_roi = y0 + int(1.2 *(y1 -y0))
            x_margin = ((y1_roi - y0) - (x1 - x0))//2
            x0_roi = x0 - x_margin
            x1_roi = x1 + x_margin
            if x0_roi < 0 or x1_roi > w or y0 < 0 or y1_roi > h:
                continue
            age_gender_roi = frame[y0:y1_roi,x0_roi:x1_roi]
            scaled_age_gender_roi = cv2.resize(age_gender_roi,age_gender_blob_size,interpolation=cv2.INTER_LINEAR).astype(np.float32)
            scaled_age_gender_roi[:] -= age_gender_average_image
            age_gender_blob = cv2.dnn.blobFromImage(scaled_age_gender_roi,size=age_gender_blob_size)
            age_model.setInput(age_gender_blob)
            age_results = age_model.forward()
            age_id = np.argmax(age_results)
            age_label = age_labels[age_id]
            age_confidence = age_results[0,age_id]
            
            gender_model.setInput(age_gender_blob)
            gender_results = gender_model.forward()
            gender_id = np.argmax(gender_results)
            gender_label = gender_labels[gender_id]
            gender_confidence = gender_results[0,gender_id]
            
            cv2.rectangle(frame, (x0,y0), (x1,y1), (255,0,0),2)
            cv2.rectangle(frame, (x0_roi,y0),(x1_roi,y1_roi),(0,255,255),2)
            text = "%s years (%.1f%%), %s (%.1f%%" % (age_label,age_confidence * 100.0,gender_label,gender_confidence * 100.0)
            cv2.putText(frame, text, (x0_roi,y0 - 20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    
    cv2.imshow("Faces,age,gender", frame)
    if (cv2.waitKey(1) == ord('q')):
        break

    succes,frame = cap.read()

cv2.destroyAllWindows()
    

