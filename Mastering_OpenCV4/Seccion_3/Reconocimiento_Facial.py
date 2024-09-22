# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:25:01 2021

@author: MBI
"""
import cv2,dlib,cvlib,face_recognition
import numpy as np
import matplotlib.pyplot as plt
#%%
"""
Introduccion al procesamiento de rostros.

Esta tecnica esta compuesta de varios pasos, que se muestran a continuacion:
    .Deteccion de marcas faciales, es un caso especifico de deteccion donde la tarea es localizar las marcas 
    importantes en un rostro.
    .Rastreo de rostros es un caso especifico del rastreo de objetos, donde la tarea es encontrar ambos, la localizacion
    y el tamaño  de todos los movimientos de rostros en un video por el rastreo hacia un conteo de informacion extra que 
    pude ser extraida de los frame de un video.
    .Reconocimiento facial es in caso especifico del reconocimiento de objetos, donde una persona es identificada o verificada
    de una imagen digital o video usando la informacion extraida de el rostro:
        .Identificacion facial (1:N): La tarea es encontrar el cuadro mas cerrado de la ejecucion de una persona desconocida en 
        coneccion con rostros conocidos.
        .Verificacion facial (1:1): La tarea es chequear  si la persona es quien clama ser.
        
"""
#%%
img = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/test_face_detection.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cas_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
cas_default = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


faces_alt2 = cas_alt2.detectMultiScale(gray)
faces_default = cas_default.detectMultiScale(gray)

# Variante para la deteccion:
retval,face_haar_alt2 = cv2.face.getFacesHAAR(img,cv2.data.haarcascades +"haarcascade_frontalface_alt2.xml")
retval,face_haar_default = cv2.face.getFacesHAAR(img,cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Salida cas_alt2: {} - {}\n".format(faces_alt2,faces_alt2.shape))
print("Salida de cas_default: {} - {}\n".format(faces_default,faces_default.shape))
print("Salida de face_haar_alt2: {} - {}\n".format(face_haar_alt2,face_haar_alt2.shape))
print("Salida de face_haar_default: {} - {}\n".format(face_haar_default,face_haar_default.shape))

face_haar_alt2 = np.squeeze(face_haar_alt2)
face_haar_default = np.squeeze(face_haar_default)

def show_detection(image,faces):
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),5)
    return image

image_faces_alt2 = show_detection(img.copy(),faces_alt2)
image_faces_defalut = show_detection(img.copy(),faces_default)
image_faces_haar_alt2 = show_detection(img.copy(),face_haar_alt2)
image_faces_haar_default = show_detection(img.copy(),face_haar_default)

fig = plt.figure(figsize=(12,5))
plt.suptitle('Face detection using haar feature-based cascade classifiers',fontsize=14,fontweight='bold')

plt.subplot(221)
plt.imshow(image_faces_alt2[:,:,::-1])
plt.title('Faces alt2')
plt.axis('off')
plt.subplot(222)
plt.imshow(image_faces_defalut[:,:,::-1])
plt.title('Face default')
plt.axis('off')
plt.subplot(223)
plt.imshow(image_faces_haar_alt2[:,:,::-1])
plt.title('Faces haar alt2')
plt.axis('off')
plt.subplot(224)
plt.imshow(image_faces_haar_default[:,:,::-1])
plt.title('Faces haar defalult')
plt.axis('off')
plt.show()
#%%
# Uando detectores de caras de gatos

cat = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/test_cat_face_detection.jpg')
gray_cat = cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)


cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
cas_ext = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")

cat_ = cas.detectMultiScale(gray_cat)
cat_ext = cas_ext.detectMultiScale(gray_cat)

retval,cat_haar = cv2.face.getFacesHAAR(cat,cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
retval,cat_haar_ext = cv2.face.getFacesHAAR(cat,cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")

cat_haar = np.squeeze(cat_haar)
cat_haar_ext = np.squeeze(cat_haar_ext)

image_cat = show_detection(cat.copy(),cat_)
image_cat_ext = show_detection(cat.copy(),cat_ext)
image_cat_haar = show_detection(cat.copy(),cat_haar)
image_cat_haar_ext = show_detection(cat.copy(),cat_haar_ext)

fig = plt.figure(figsize=(12,5))
plt.suptitle('Face detection using haar feature-based cascade classifiers',fontsize=14,fontweight='bold')


plt.subplot(221)
plt.imshow(image_cat[:,:,::-1])
plt.title('Cat')
plt.axis('off')
plt.subplot(222)
plt.imshow(image_cat_ext[:,:,::-1])
plt.title('Cat extended')
plt.axis('off')
plt.subplot(223)
plt.imshow(image_cat_haar[:,:,::-1])
plt.title('Cat haar')
plt.axis('off')
plt.subplot(224)
plt.imshow(image_cat_haar_ext[:,:,::-1])
plt.title('Cat haar extended')
plt.axis('off')
plt.show()
#%%
# Usando los modulos de dnn

"""
Opencv provee el modulo cv2.dnn para el uso de redes neuronales profundas.
Este modulo dnn implementa la inferencia hacia adelante con redes  pre entrenadas usando
framework  populares como  Caffe,TensorFlow,Torch y Darknet.
Opencv provee dos modelos para la deteccion:
    .Face detector (FP16): Version de 16 puntos flotantes original de la implementacion Caffe (5.1 MB)
    .Face detector (UINT8): Version cunatizada de 8-bit que usa TensorFlow (2.6 MB)
    

Para el modelo Coffe:
    .rs_300x300_ssd_iter_140000_fp16.caffemodel: Este archivo contiene los pesos para las capas actuales.
    .deploy.prototxt: Este es el archivo de la arquitectura del modelo.

Para el model TensorFlow:
    .opencv_face_detector_uint8.pb: Este archivo contiene los pesos para las capas actuales
    .opencv_face_detector.pbtxt: Este archivo tiene la arquitectura  del modelo.

"""
#%%
image = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/test_face_detection.jpg')
(h,w) = image.shape[:2]
image_copy = image.copy()
net = cv2.dnn.readNetFromCaffe('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/deploy.prototxt','/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/res10_300x300_ssd_iter_140000_fp16.caffemodel')
blob = cv2.dnn.blobFromImage(image,1.0,(300,300),[104.,117.,123.],False,False)
# Para arquirir la mejor presicion se deve correr el modelo con imagenes bgr de tamaño 300x300 y aplicar una sustraccion a los valores bgr de [104.,117.,123.]

print("Shape blob: {}\n".format(blob.shape))
net.setInput(blob)
detections = net.forward()
print("Shape detections: {}".format(detections.shape))
detected_face = 0
for i in range(0,detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > 0.7:
        detected_face += 1
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype('int')
        
        text = "{:.3f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image_copy,(startX,startY),(endX,endY),(255,0,0),3)
        cv2.putText(image_copy,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
        
fig  = plt.figure(figsize=(10,5))
plt.suptitle("Face detection using OpneCV dnn face detector",fontsize=14,fontweight='bold')

plt.imshow(image_copy[:,:,::-1])
plt.title("DNN face detector Coffe: {}".format(detected_face))
plt.axis('off')
plt.show()

#%%
# Usando TensorFlow

net = cv2.dnn.readNetFromTensorflow('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/opencv_face_detector_uint8.pb','/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/opencv_face_detector.pbtxt')
blob = cv2.dnn.blobFromImage(image,1.0,(300,300),[104.,117.,123.],False,False)
net.setInput(blob)
detections = net.forward()

detected_face = 0
for i in range(0,detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > 0.7:
        detected_face += 1
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype('int')
        
        text = "{:.3f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image,(startX,startY),(endX,endY),(255,0,0),3)
        cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

fig  = plt.figure(figsize=(10,5))
plt.suptitle("Face detection using OpneCV dnn face detector",fontsize=14,fontweight='bold')

plt.imshow(image_copy[:,:,::-1])
plt.title("DNN face detector TensorFlow: {}".format(detected_face))
plt.axis('off')
plt.show()
#%%
# Deteccion facial con dlib
image = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/test_face_detection.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
rects_1 = detector(gray,0) # El segundo argumento indica la cantidad de veces que se muestrea la imagen en gris
rects_2 = detector(gray,1)

def show_detections(image,faces):
    
    for i in faces:
        cv2.rectangle(image,(i.left(),i.top()),(i.right(),i.bottom()),(255,0,0),10)
    return image

detection_1 = show_detections(image.copy(),rects_1)
detection_2 = show_detections(image.copy(),rects_2)

fig = plt.figure(figsize=(10,5))
plt.suptitle("Face detection using dlib frontal detector",fontsize=14,fontweight='bold')

plt.subplot(121)
plt.imshow(detection_1[:,:,::-1])
plt.title('Detection_1 gray image: {} '.format(len(rects_1)))
plt.axis('off')
plt.subplot(122)
plt.imshow(detection_2[:,:,::-1])
plt.title("Detection_2 gray image: {}".format(len(rects_2)))
plt.axis('off')
plt.show()

#%%
#  Usando dlib.cnn_face_detection_model_v1()

cnn_face_detector = dlib.cnn_face_detection_model_v1("/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/mmod_human_face_detector.dat")
rects = cnn_face_detector(image,0)

def show_detections_dlib(image,faces):
    for face in faces:
        cv2.rectangle(image,(face.rect.left(),face.rect.top()),(face.rect.right(),face.rect.bottom()),(255,0,0),10)
    return image 

image_faces = show_detections_dlib(image,rects)

fig = plt.figure(figsize=(10,5))
plt.suptitle("Face detection using dlib CNN detector",fontsize=14,fontweight='bold')

plt.imshow(image_faces[:,:,::-1])
plt.title("CNN_FACE_DETECTOR(img,0): {}".format(len(rects)))
plt.axis('off')
plt.show()


#%%
# Deteccion de rostros con face_recognition
image = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/test_face_detection.jpg')
rects_1 = face_recognition.face_locations(image,0,model="hog")
rects_2 = face_recognition.face_locations(image,1,model="hog")

def show_detection_fr(image,faces):
    for face in faces:
        top,right,bottom,left = face 
        cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),10)
    return image

img_1 = show_detection_fr(image.copy(),rects_1)
img_2 = show_detection_fr(image.copy(),rects_2)

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('silver')
plt.suptitle('Face recognition with face_recognition hog',fontsize=14,fontweight='bold')
plt.subplot(121)
plt.imshow(img_1[:,:,::-1])
plt.title("Detection with 0 upsampling: {}".format(len(rects_1)))
plt.axis('off')
plt.subplot(122)
plt.imshow(img_2[:,:,::-1])
plt.title('Detection with 1 upsampling: {}'.format(len(rects_2)))
plt.axis('off')
plt.show()

#%%

cnn_1 = face_recognition.face_locations(image,0,"cnn")
cnn_2 = face_recognition.face_locations(image,1,"cnn")

img_cnn_1 = show_detection_fr(image.copy(), cnn_1)
img_cnn_2 = show_detection_fr(image.copy(),cnn_2)

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('silver')
plt.suptitle('Face recognition with face_recognition cnn',fontsize=14,fontweight='bold')
plt.subplot(121)
plt.imshow(img_cnn_1[:,:,::-1])
plt.title("Detection with 0 upsampling: {}".format(len(cnn_1)))
plt.axis('off')
plt.subplot(122)
plt.imshow(img_cnn_2[:,:,::-1])
plt.title('Detection with 1 upsampling: {}'.format(len(cnn_2)))
plt.axis('off')
plt.show()

#%%
# Usando cvlib para la deteccion

faces,confidence = cvlib.detect_face(image,enable_gpu=True)

def show_detection_cvlib(img,faces,confidence):
    for (startx,starty,endx,endy),i in zip(faces,confidence):
        cv2.rectangle(img,(startx,starty),(endx,endy),(255,0,0),5)
        text = "{:.3f}%".format(i * 100)
        y = starty - 10 if starty > 10 else starty + 10
        cv2.putText(img,text,(startx,y),cv2.FONT_HERSHEY_SIMPLEX,0.99,(0,0,255),2)
    
    return img 

print("Confidence: ",confidence)
cvlib_image = show_detection_cvlib(image.copy(),faces,confidence)

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('silver')
plt.suptitle('Face recognition with cvlib',fontsize=14,fontweight='bold')

plt.imshow(cvlib_image[:,:,::-1])
plt.title("Cvlib detection")
plt.axis('off')
plt.show()

#%%