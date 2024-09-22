# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 20:17:54 2021

@author: MBI
"""
import cv2,os
import numpy as np
#%%
# Detectar caras sobre una imagen

image = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo5/Learning OpenCV 4 Computer Vision with Python 3_page364_image59.jpg")

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = face_detector.detectMultiScale(gray_image,2.00,5)

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

cv2.namedWindow("Woodcutters Detected")
cv2.imshow("Woodcutters Detected",image)

while True:
    if cv2.waitKey(-1) & 0xff == ord('q'):
        break

cv2.destroyWindow("Woodcutters Detected")

#%%
# Detectar caras sobre un video

face_video_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

camera = cv2.VideoCapture(0)

while (cv2.waitKey(1) == -1):
    success,frame = camera.read()
    if success:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_video_detector.detectMultiScale(gray,1.2,5,minSize=(120,120))
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
            roi_gray = gray[y:y+h,x:x+w]
            eyes = eyes_cascade.detectMultiScale(roi_gray,1.2,5,minSize=(30,30))
            for ex,ey,ew,eh in eyes:
                cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),1)
    
        cv2.imshow("Face Detection",frame)
    

cv2.destroyAllWindows()
#%%
# Entrenamiento de un clasificador persnonalizado

output_folder_maikel = "C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo5/Maikel"
output_folder_maria = "C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo5/Maria"

if not os.path.exists(output_folder_maikel) and  not os.path.exists(output_folder_maria):
    os.mkdir(output_folder_maikel)
    os.mkdir(output_folder_maria)
    print("Directorio creado")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

trainin_image_size = (200,200)
camera = cv2.VideoCapture(0)

def preprocess_image(directorio,detector,device,count=0):
    while (cv2.waitKey(1) == -1):
        succes,frame = device.read()
        if succes:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray,1.2,5,minSize=(120,120))
            for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
                face_img = cv2.resize(gray[y:y+h,x:x+w],(200,200))
                face_filename = "%s/%d.pgm"%(directorio,count)
                cv2.imwrite(face_filename,face_img)
                count += 1
                cv2.imshow("Capturing Faces...",frame)
    
    cv2.destroyAllWindows()


            
valor = input("Presione E para procesar imagenes (Maikel): ")

if valor == "E":
    
    preprocess_image(output_folder_maikel,face_cascade,camera)
    print("Procesamiento terminado para Maikel")
    
    valor = input("Presione E para procesar imagenes (Maria): ")
    
    preprocess_image(output_folder_maria,face_cascade,camera)
    print("Procesameineto terminado par Maria")
    

# Training the model

def load_image(path,imega_size):
    names = []
    training_images,training_labels = [],[]
    label = 0
    for dirname,subdirnames,filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname,subdirname)
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path,filename),cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img,imega_size)
                training_images.append(img)
                training_labels.append(label)
            label += 1
    
    training_images = np.asarray(training_images,np.uint8)
    training_labels = np.asarray(training_labels,np.int32)
    return names,training_images,training_labels


names,data,labes = load_image("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo5",trainin_image_size)

print("Imagenes cargadas.")

model = cv2.face.EigenFaceRecognizer_create()
model.train(data,labes)



while (cv2.waitKey(1) == -1):
    success,frame = camera.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if success:
        faces = face_cascade.detectMultiScale(gray,1.2,5)
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
            roi_gray = gray[x:x+w,y:y+h]
            if roi_gray.size == 0:
                continue
            roi_gray = cv2.resize(roi_gray,trainin_image_size)
            label,confidence = model.predict(roi_gray)
            text = "%s, confidence = %.2f" % (names[label],confidence)
            cv2.putText(frame,text,(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow("Face Recognition",frame)

cv2.destroyAllWindows()
#%%

