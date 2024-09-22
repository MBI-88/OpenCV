# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:32:47 2021

@author: MBI
"""
import cv2,dlib,face_recognition
import numpy as np
import matplotlib.pyplot as plt
#%%
# Rastreo de rostros

def draw_text_info():
    menu_pos_1 = (10,20)
    menu_pos_2 = (10,40)
    
    cv2.putText(frame,"[*]Use  '1' to re-initialize tracking",menu_pos_1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    if tracking_face:
        cv2.putText(frame,"[*]Tracking the face",menu_pos_2,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
    else:
        cv2.putText(frame,"[*]Detencting a face to initialize tracking...",menu_pos_2,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))

capture = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
tracker = dlib.correlation_tracker() # Valores recomendados son: filter_size (5,6,7),num_scale (4,5,6).Nota: Valores  mas altos dan mejor presicion  pero mayor consumo de recursor de computo.
tracking_face = False

while True:
    ret,frame = capture.read()
    draw_text_info()
    
    if tracking_face is False:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        rects = detector(gray,0)
        if len(rects) > 0:
            tracker.start_track(frame,rects[0])
            tracking_face = True
    
    if tracking_face is True:
        print(tracker.update(frame))
        pos = tracker.get_position()
        cv2.rectangle(frame,(int(pos.left()),int(pos.top())),(int(pos.right()),int(pos.bottom())),(0,255,0),3)
    
    key = 0xFF & cv2.waitKey(1)
    if key == ord("1"):
        tracking_face = False
    if key == ord("q"):
        break
    cv2.imshow("[*]Face tracking using dlib frontal detector and correlation filters for tracking",frame)

capture.release()
cv2.destroyAllWindows()

#%%
# Detectando objetos con dlib (DCF-based tracker)

def draw_text_info():
    menu_pos_1 = (10,20)
    menu_pos_2 = (10,40)
    menu_pos_3 = (10,60)
    
    info_1 = "[*]Use left click of the mouse to select the object to track"
    info_2 = "[*]Use '1' to start tracking, '2' to reset tracking and 'q' to exit"
    
    cv2.putText(frame,info_1,menu_pos_1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    cv2.putText(frame,info_2,menu_pos_2,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    if tracking_state:
        cv2.putText(frame,"[*]Tracking...",menu_pos_3,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
    else:
        cv2.putText(frame,"[-]Not tracking",menu_pos_3,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))

points = []

def mouse_event_handler(event,x,y,flags,param):
    global points
    if event  == cv2.EVENT_LBUTTONDOWN:
        points = [(x,y)]
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x,y))

capture = cv2.VideoCapture(0)

window_name = "Object tracking using dlib correlation filter algoritmo"

cv2.namedWindow(window_name)

cv2.setMouseCallback(window_name,mouse_event_handler)

tracker = dlib.correlation_tracker()
tracking_state = False

while True:
    ret,frame = capture.read()
    draw_text_info()
    if len(points) == 2:
        cv2.rectangle(frame,points[0],points[1],(0,0,255),3)
        dlib_rectangle = dlib.rectangle(points[0][0],points[0][1],points[1][0],points[1][1])
    
    if tracking_state == True:
        tracker.update(frame)
        pos = tracker.get_position()
        cv2.rectangle(frame,(int(pos.left()),int(pos.top())),(int(pos.right()),int(pos.bottom())),(0,255,0),3)
    
    key = 0xFF & cv2.waitKey(1)
    if key == ord("1"):
        if len(points) == 2:
            tracker.start_track(frame,dlib_rectangle)
            tracking_state = True
            points = []
    
    if key == ord("2"):
        points = []
        tracking_state = False
    
    if key == ord("q"):
        break

    cv2.imshow(window_name,frame)

capture.release()
cv2.destroyAllWindows()
#%%
# Reconocimiento facial con OpneCV

"""
OpenCv propone soprte para reconosimineto de rostros con los siguientes algoritmos:
    .Eigenfaces
    .Fisherfaces
    .Local Binary Patterns Histograms (LBPH)

Estas implementaciones interactuan con el reconosimineto en diferentes formas.
Ejemplo:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer = cv2.face.FisherFaceRecognizer_create()

Estos algoritmos hacen uso de los metodos train() y predict().
El modelo LBPH deveria proveer el mejor resultado de los tres metodos. El LBPH soporta el metodo update() el cual se usa para  hacer un nuevo entrenamiento del modelo en un nuvo set de rostros. Los otros dos modelos no  soportan este metodo.

Finalmente estos modelos proponen los metodos write() y read() que se usan para salvar y cargar un modelo previamente entrenado.

"""
#%%
# Reconocimineto facial con dlib

def face_encodigs(face_image,number_of_time_to_sample=1,num_jitter=1):
    face_locations = detector(face_image,number_of_time_to_sample)
    raw_landmarks = [pose_prediction_5_points(face_image,face_location) for face_location in face_locations]
    
    return [np.array(face_encoder.compute_face_descriptor(face_image,raw_landmarks_set,num_jitter))for raw_landmarks_set in raw_landmarks]


pose_prediction_5_points = dlib.shape_predictor("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/dlib_face_recognition_resnet_model_v1.dat")

detector = dlib.get_frontal_face_detector()

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/jared_1.jpg')

rgb = image[:,:,::-1]

encoding = face_encodigs(rgb)
print(encoding[0],"\n")

def compare_faces(face_encoding,encoding_to_check):
    return list(np.linalg.norm(face_encoding - encoding_to_check,axis=1))
    
def compare_faces_ordered(face_encoding,face_names,encoding_to_check):
    distance = list(np.linalg.norm(face_encoding - encoding_to_check,axis=1))
    return zip(*sorted(zip(distance,face_names)))


known_image_1 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/jared_1.jpg')
known_image_2 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/jared_2.jpg')   
known_image_3 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/jared_3.jpg')
known_image_4 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/obama.jpg')
uknown_image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/jared_4.jpg')

known_image_1 = known_image_1[:,:,::-1]
known_image_2 = known_image_2[:,:,::-1]
known_image_3 = known_image_3[:,:,::-1]
known_image_4 = known_image_4[:,:,::-1]
uknown_image = uknown_image[:,:,::-1]

names = ["jared_1.jpg","jared_2.jpg","jared_3.jpg","obama.jpg"]

known_image_1_encodig = face_encodigs(known_image_1)[0]
known_image_2_encodig = face_encodigs(known_image_2)[0]
known_image_3_encodig = face_encodigs(known_image_3)[0]
known_image_4_encodig = face_encodigs(known_image_4)[0]
known_encoding = [known_image_1_encodig,known_image_2_encodig,known_image_3_encodig,known_image_4_encodig]

unkown_encoding = face_encodigs(uknown_image)[0]

computed_distance_ordered,ordered_names = compare_faces_ordered(known_encoding,names,unkown_encoding)
computed_distance = compare_faces(known_encoding,unkown_encoding)
print("Distancia computad: {}\n".format(computed_distance))
print("Distancia computada ordenada: {}\n".format(computed_distance_ordered))
print("Nombres ordenados: {}".format(ordered_names))

# Nota: si dos de las imagenes usadas tiene un distancia euclidiana menor a 0.6 se consideran similares.

fig = plt.figure(figsize=(10,8))
fig.patch.set_facecolor('silver')
plt.suptitle("Images comparations",fontsize=14,fontweight='bold')

plt.subplot(241)
plt.imshow(cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/'+ ordered_names[0])[:,:,::-1])
plt.title("{}\n {}".format(computed_distance_ordered[0],ordered_names[0]))
plt.axis('off')
plt.subplot(242)
plt.imshow(cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/'+ordered_names[1])[:,:,::-1])
plt.title("{}\n {}".format(computed_distance_ordered[1],ordered_names[1]))
plt.axis('off')
plt.subplot(243)
plt.imshow(cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/'+ordered_names[2])[:,:,::-1])
plt.title("{}\n {}".format(computed_distance_ordered[2],ordered_names[2]))
plt.axis('off')
plt.subplot(244)
plt.imshow(cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/'+ordered_names[3])[:,:,::-1])
plt.title("{}\n {}".format(computed_distance_ordered[3],ordered_names[3]))
plt.axis('off')
plt.subplot(245)
plt.imshow(uknown_image)
plt.title("jared_4.jpg")
plt.axis('off')
plt.show()
#%%
# Reconocimiento facial con face_recognition

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/jared_1.jpg')

image = image[:,:,::-1]
encoding = face_recognition.face_encodings(image)

print(encoding[0], "\n")


known_image_1 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/jared_1.jpg')
known_image_2 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/jared_2.jpg')   
known_image_3 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/jared_3.jpg')
known_image_4 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/obama.jpg')

names = ["jared_1.jpg","jared_2.jpg","jared_3.jpg","obama.jpg"]

unkown_image = face_recognition.load_image_file('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/jared_4.jpg')

known_image_1_encodig = face_recognition.face_encodings(known_image_1)[0]
known_image_2_encodig = face_recognition.face_encodings(known_image_2)[0]
known_image_3_encodig = face_recognition.face_encodings(known_image_3)[0]
known_image_4_encodig = face_recognition.face_encodings(known_image_4)[0]

known_encoding = [known_image_1_encodig,known_image_2_encodig,known_image_3_encodig,known_image_4_encodig]

unknown_encoding = face_recognition.face_encodings(unkown_image)[0]

result = face_recognition.compare_faces(known_encoding,unknown_encoding)

print(result)

fig = plt.figure(figsize=(10,6))
fig.patch.set_facecolor('silver')
plt.suptitle("Images comparations using face_recognition",fontsize=14,fontweight='bold')

for i,(x,y) in enumerate(zip(result,names)):
    plt.subplot(2,4,i+1)
    plt.imshow(cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/'+y)[:,:,::-1])
    plt.title("Check: {}".format(x))
    plt.axis('off')

plt.subplot(245)
plt.imshow(unkown_image)
plt.title("Unkown image")
plt.axis('off')
plt.show()

#%%


    
    
    
