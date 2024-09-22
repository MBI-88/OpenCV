# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:30:54 2021

@author: MBI
"""
import cv2 
import  matplotlib.pyplot as plt
import numpy as np
#%%
"""
Introduccion a realidad argumentada.

La realidad aumentada tiene 2 importantestipos de aumentacion: basado en locacion y basdo en recono-
simiento. Ambos tipos tratan de encontrar donde el usuario esta observando. Esta informacion es la llave en
el proceso de realidad argumentada,y la confienza en el calculo de la posicion de la camara.

Basada en locacion :
    Confia en detectar la loacion y orientacion del usuario por la lectura de los sensores 
    en el dispositivo. Esta informacion es usada para superponer los elementos de realidad aumentada gene-
    rados por  la computadora en la pantalla.

Basada en reconocimiento :
    Usa tecnicas de procesado de imagenes para conocer donde el usuario esta observando. Obteniendo la posicion
    de la camara de una imagen necesita encontrar la correspodiente proyeccion de la camara. Para encontrar estas 
    correspodencias 2 importantes acercamientos pueden ser encontrados en la literatura:
        
        Estimacion  basada en marcas: 
            Este acercamiento confia en el uso de marcas planares para computar la posicion
        de la camara de sus 4 esquinas. Su mayor desventaja es el uso de marcas cuadradas  que estan en coneccion con 
        la posicion de la camara, lo cual confia en la presicion de la determinacion de las 4 esquinas de la marca.
        
        Estimacion basada en el no uso  de marcas :
            Cuando la esena no  puede ser preparada usando marcas para conocer la posicion de la  estimacion , el objeto
            es naturalmente presentado en la imagen para estimar la posicion. En un set de 2D puntos y su correspondiente
            3D cordenadas ha sido calculado para resolver el problema de la estimacion de la posicion de la camara.
"""
# Deteccion de variables en una camara
"""
Algoritmos que ofrece opencv para la deteccion de variables en una camara:
    Harris Corner Detection
    Shi-Tomasi Corner Detection
    Scale Invariant Feature Transform (SIFT)
    Speeded-Up Robust Features (SURF)
    Features from Accelerated Segment Test (FAST)
    Binary Robust Independent Elementary Features (BRIEF)
    Oriented FAST and Rotated BRIEF (ORB)
"""
#%%
image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/opencv_logo_with_text.png')
orb = cv2.ORB_create()
keypoints = orb.detect(image,None) # se puede usar tambien orb.detectAndCompute hace ambas operaciones, devuelve 2 parametros
keypoints,descriptors = orb.compute(image, keypoints)

imag_points = cv2.drawKeypoints(image,keypoints,None,color=(255,0,255),flags=0)

fig = plt.figure(figsize=(10,8))
fig.patch.set_facecolor('silver')
plt.suptitle('ORB keypoint detector',fontsize=16,fontweight='bold')
plt.subplot(121)
plt.imshow(image)
plt.title('Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(imag_points)
plt.title('Detected keypoints')
plt.axis('off')
plt.show()

#%%
# Ejecutando variables
"""
OpenCV provee dos formas de ejecutar variables: 
    Fuerza bruta (BF): 
        Este ejecutor toma cada uno de los descriptores compudatos para cada
    uno de las variables detectadas en el primer set y es ejecutado con todos los otros descriptores 
    en el segundo set. Finalmente devuelve la ejecucion con la disancia mas corta.
    
    Libreria rapida para aproximacion de vecinos mas cercanos (FLANN):
        Este ejecutor trabaja mas rapido que el BF para grandes dataset. Contiene algoritmos de optimizacion
        para encontrar vecinos mas cercanos.
    
    
Descripcion de cv2.BFMatcher():
    normType: 
        Establece la medida de la distancia para usar , por defecto es cv2.NORM_L2.
    En caso de uar ORB o otros  descriptores basados en descripcion binaria como BRIEF o BRISK, la medida de la
    distancia para usar es cv2.NORM_HAMMING.
    
    crossCheck:
        El cual es por defecto falso, puede ser establecido a verdadero  para que devuelva solamente los pares con-
        sistentes en el proceso  de ejecucion (las 2 variables en ambos set deveran ejecutarse una con la otra)
"""
#%%
image_scence = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/opencv_logo_with_text_scene.png')
image_query = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/opencv_logo_with_text.png')
orb = cv2.ORB_create()
keypoints_1,descriptor_1 = orb.detectAndCompute(image_query,None)
keypoints_2,descriptor_2 = orb.detectAndCompute(image_scence,None)

bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

bf_matches = bf_matcher.match(descriptor_1,descriptor_2)

bf_matches = sorted(bf_matches,key=lambda x:x.distance)

result = cv2.drawMatches(image_query,keypoints_1,image_scence,keypoints_2,bf_matches[:20],None,matchColor=(255,255,0),singlePointColor=(255,0,255),flags=0)

fig = plt.figure(figsize=(8,6))
fig.patch.set_facecolor('silver')
plt.suptitle('ORB descriptor and Brute-Force (BF) matcher',fontsize=16,fontweight='bold')
plt.imshow(result[:,:,::-1])
plt.title('Matches between the two images')
plt.axis('off')
plt.show()

#%%
# Ejecucion de variables y computacion de homografia para encontrar objetos
"""
Ejecucion de variables y computacion de la homografia para encotrar objetos

OpenCV provee varios  metodos para  computar la matrix de homografia-RANSAC,Least-Median (LMEDS)
y PROSAC (RHO). 
"""
#%%
image_query = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/opencv_logo_with_text.png')
image_scence = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/opencv_logo_with_text_scene.png')
orb = cv2.ORB_create()
keypoints_1,descriptor_1 = orb.detectAndCompute(image_query,None)
keypoints_2,descriptor_2 = orb.detectAndCompute(image_scence,None)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

bf_matches = bf_matcher.match(descriptor_1,descriptor_2)

bf_matches = sorted(bf_matches,key=lambda x: x.distance)
best_matches = bf_matches[:40]

pts_src = np.float32([keypoints_1[m.queryIdx].pt for m in best_matches]).reshape(-1,1,2)
pts_dst = np.float32([keypoints_2[m.trainIdx].pt for m in best_matches]).reshape(-1,1,2)


M,mask = cv2.findHomography(pts_src,pts_dst,cv2.RANSAC,5.0)#  El valor de 5.0 establece el maximo error en la reproyeccion de puntos
#para valores superiores a 5.0 se consideran outlier.

h,w = image_query.shape[:2]
pts_corners_src = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

pts_corner_dst = cv2.perspectiveTransform(pts_corners_src,M)

img_obj = cv2.polylines(image_scence,[np.int32(pts_corner_dst)],True,(0,255,255),10)

img_matching = cv2.drawMatches(image_query,keypoints_1,img_obj,keypoints_2,best_matches,None,matchColor=(255,255,0),singlePointColor=(255,0,255),flags=0)

fig = plt.figure(figsize=(8,6))
fig.patch.set_facecolor('silver')
plt.suptitle('Feature matching & homography computation',fontsize=16,fontweight='bold')
plt.imshow(img_matching[:,:,::-1])
plt.title('feature matching')
plt.axis('off')
plt.show()

#%%
# Realidad argumentada basada en marcas
"""
Ceracion de marcadores y diccionarios

Marcador Aruco es un marcador cuadrado compuesto de celdas internas y externas llamada bits. Las celdas externas son el set 
para el negro, creando un borde  externo que puede ser rapido y robusto en la deteccion
La celdas internas son usadas para la codificacion de marcas. El marcador Aruco puede ser creado con diferentes tamaños.
El tamaño de las marcas indica el numero de celdas internas relacionadas con la matrix interna.
Un diccionario de marcadores es el set de marcadores considerado para ser usado en una app especifica.


"""
#%%
aruco_diccionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

aruco_marker_1 = cv2.aruco.drawMarker(dictionary=aruco_diccionary,id=2,sidePixels=600,borderBits=1)
aruco_marker_2 = cv2.aruco.drawMarker(dictionary=aruco_diccionary,id=2,sidePixels=600,borderBits=2)
aruco_marker_3 = cv2.aruco.drawMarker(dictionary=aruco_diccionary,id=2,sidePixels=600,borderBits=3)

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('silver')
plt.suptitle('Aruco markers cration',fontsize=16,fontweight='bold')
plt.subplot(131)
plt.imshow(aruco_marker_1,cmap='gray')
plt.title('marker_DICT_7x7_250_600_1')
plt.axis('off')
plt.subplot(132)
plt.imshow(aruco_marker_2,cmap='gray')
plt.title('marker_DICT_7x7_250_600_2')
plt.axis('off')
plt.subplot(133)
plt.imshow(aruco_marker_3,cmap='gray')
plt.title('marker_DICT_7x7_250_600_3')
plt.axis('off')
plt.show()

#%%
# Detectando marcadores
parameters = cv2.aruco.DetectorParameters_create()
capture = cv2.VideoCapture(0)
while True:
    ret,frame = capture.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    corners,ids,rejected_corners = cv2.aruco.detectMarkers(gray_frame,aruco_diccionary,parameters=parameters)
    frame = cv2.aruco.drawDetectedMarkers(image=frame,corners=corners,ids=ids,borderColor=(0,255,0))
    frame = cv2.aruco.drawDetectedMarkers(image=frame,corners=rejected_corners,borderColor=(0,0,255))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

#%%
# Calibracion de camara
"""
OpenCv provee la funcion cv2.aruco.calibrateCameraCharuco()
Parametros de la funcion:
    
    charucoCorners: Vector que contiene las esquinas charuco
    charucoIds: Lista de identificadores,representa la tabla.
    imageSize: El tamaño de imagen de entrada.
    rvecs: Vector de salida,contiene  un vector de de la rotacion estimada  para cada una de las tablas
    tvecs: Es el vector de  traslacion de vectores estimado para cada una  de las partes vistas.
    cameraMatrix: Matriz de la camara
    distCoeffs: Coeficiente de distorcion
        
"""
#%%
import pickle

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
board = cv2.aruco.CharucoBoard_create(3,3,.025,.0125,dictionary)
image_board = board.draw((200*3,200*3))

cv2.imwrite('Charuco.png',image_board)

cap = cv2.VideoCapture(0)
all_corners = []
all_ids = []
counter = 0
for i in range(300):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray,dictionary)
    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
        if res2[1] is not None and res2[2] is not None and len(res2[1] > 3 and counter % 3 == 0):
            all_corners.append(res2[1])
            all_ids.append(res2[2])
        cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    counter += 1

try :
    cal = cv2.aruco.calibrateCameraCharuco(all_corners,all_ids,board,gray.shape,None,None)
except :
    cap.release()
    print('Calibration could not be done...')
    cv2.destroyAllWindows()

retval,cameraMatrix,distCoeffs,rvecs,tvecs = cal
f = open('Calibration_1.pckl','wb')
pickle.dump((cameraMatrix,distCoeffs),f)
f.close()

cap.release()
cv2.destroyAllWindows()
#%%
# Estimacion de la posicion de la camara
import os 

if not os.path.exists('calibration.pckl'):
    print('You  need to calibrate the camera')
    exit()
else:
    f = open('calibration.pckl','rb')
    cameraMatrix,distCoeffs = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print('Calibration issue. Remove ./calibartion.pckl and recalibrate your camera')
        exit()

aruco_diccionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
parameters = cv2.aruco.DetectorParameters_create()
capture = cv2.VideoCapture(0)

while True:
    ret,frame = capture.read()
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    corners,ids,rejectedImgPoints = cv2.aruco.detectMarkers(gray_frame,aruco_diccionary,parameters=parameters)
    
    frame = cv2.aruco.drawDetectedMarkers(image=frame,corners=corners,ids=ids,borderColor=(0,255,0))
    frame = cv2.aruco.drawDetectedMarkers(image=frame,corners=rejectedImgPoints,borderColor=(0,0,255))
    
    if ids is not None:
        rvecs,tvecs,_ = cv2.aruco.estimatePoseSingleMarkers(corners,1,cameraMatrix,distCoeffs)
        for rvec,tvec in zip(rvecs,tvecs):
            cv2.aruco.drawAxis(frame,cameraMatrix,distCoeffs,rvec,tvec,1)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

#%%
# Estimacion de la posicion y argumentacion basica

import os 

if not os.path.exists('calibration.pckl'):
    print('You  need to calibrate the camera')
    exit()
else:
    f = open('calibration.pckl','rb')
    cameraMatrix,distCoeffs = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print('Calibration issue. Remove ./calibartion.pckl and recalibrate your camera')
        exit()

OVERLAY_SIZE_PER = 1

def draw_points(img, pts):
    """ Draw the points in the image"""

    pts = np.int32(pts).reshape(-1, 2)

    img = cv2.drawContours(img, [pts], -1, (255, 255, 0), -3)

    for p in pts:
        cv2.circle(img, (p[0], p[1]), 5, (255, 0, 255), -1)

    return img


# Create the dictionary object and the parameters:
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
parameters = cv2.aruco.DetectorParameters_create()

# Create video capture object 'capture' to be used to capture frames from the first connected camera:
capture = cv2.VideoCapture(0)

while True:
    # Capture frame by frame from the video capture object 'capture':
    ret, frame = capture.read()

    # We convert the frame to grayscale:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers:
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)

    # Draw detected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))

    # Draw rejected markers:
    # frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejectedImgPoints, borderColor=(0, 0, 255))

    if ids is not None:
        # rvecs and tvecs are the rotation and translation vectors respectively, for each of the markers in corners.
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            # Note: The marker coordinate system is centered on the center of the marker
            # The coordinates of the four corners of the marker (in its own coordinate system) are:
            # 1: (-markerLength/2, markerLength/2, 0)
            # 2: (markerLength/2, markerLength/2, 0)
            # 3: (markerLength/2, -markerLength/2, 0)
            # 4: (-markerLength/2, -markerLength/2, 0)
            # Define the points where you want the image to be overlaid (remember: marker coordinate system):
            desired_points = np.float32(
                [[-1 / 2, 1 / 2, 0], [1 / 2, 1 / 2, 0], [1 / 2, -1 / 2, 0], [-1 / 2, -1 / 2, 0]]) * OVERLAY_SIZE_PER

            # Project the points:
            projected_desired_points, jac = cv2.projectPoints(desired_points, rvecs, tvecs, cameraMatrix, distCoeffs)

            # Draw the projected points (debugging):
            draw_points(frame, projected_desired_points)

    # Display the resulting augmented frame:
    cv2.imshow('frame', frame)

    # Press 'q' to exit:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
capture.release()
cv2.destroyAllWindows()
#%%
# Estimacion de la camara con realidad argumentada mas avanzada

def draw_argumented_overlay(pts1,overlay_image,image):
    pts2 = np.float32([[0,0],[overlay_image.shape[1],0],[overlay_image.shape[1],overlay_image.shape[0]],[0,overlay_image.shape[0]]])
    
    cv2.rectangle(overlay_image,(0,0),(overlay_image.shape[1],overlay_image.shape[0]),(255,255,0),10)
    M = cv2.getPerspectiveTransform(pts2,pts1)
    
    dst_image = cv2.warpPerspective(overlay_image,M,(image.shape[1],image.shape[0]))
    cv2.imshow('dst_image',dst_image)
    
    dst_image_gray = cv2.cvtColor(dst_image,cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(dst_image_gray,0,255,cv2.THRESH_BINARY_INV)
    
    image_masked = cv2.bitwise_and(image,image,mask=mask)
    result = cv2.add(dst_image,image_masked)
    return result 
# Esta funcion va encima  de  draw_points() dentro del ciclo while dando su retorno a frame

#%%
# Snapchat basado en realidad argumentada

moustage = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/moustache.png',-1)
face_test = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/face_test.png')
gray_face = cv2.cvtColor(face_test,cv2.COLOR_BGR2GRAY)

moustage_mask = moustage[:,:,3]
moustage = moustage[:,:,0:3]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
nose_cascade = cv2.CascadeClassifier("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/haarcascade_mcs_nose.xml")

faces = face_cascade.detectMultiScale(gray_face,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(face_test,(x,y),(x+w,y+h),(255,255,0),2)
        
    roi_gray = gray_face[y:y+h,x:x+w]
    roi_color = face_test[y:y+h,x:x+w]
        
    noses = nose_cascade.detectMultiScale(roi_gray)
    
    for (nx,ny,nw,nh) in noses:
        cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,255),2)
        
        x1 = int(nx-nw/2)
        x2 = int(nx+nw/2 + nw)
        y1 = int(ny+nh/2+nh/8)
        y2 = int(ny+nh+nh/4+nh/6)
        
        if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
            continue
        
        cv2.rectangle(roi_color,(x1,y1),(x2,y2),(255,0,0),2)
        img_moustage_res_width = int(x2-x1)
        img_moustage_res_height = int(y2-y1)
        
        mask = cv2.resize(moustage_mask,(img_moustage_res_width,img_moustage_res_height))
        mask_inv = cv2.bitwise_not(mask)
        img = cv2.resize(moustage,(img_moustage_res_width,img_moustage_res_height))
        
        roi = roi_color[y1:y2,x1:x2]
        roi_background = cv2.bitwise_and(roi,roi,mask=mask_inv)
        roi_foreground = cv2.bitwise_and(img,img,mask=mask)
        
        cv2.imshow('roi_backgraound',roi_background)
        cv2.imshow('roi_foreground',roi_foreground)
        
        res = cv2.add(roi_background,roi_foreground)
        roi_color[y1:y2,x1:x2] = res 
    
        
cv2.imshow('Snapchat-based OpenCV moustage overlay',face_test)    
cv2.waitKey(10000)
cv2.destroyAllWindows()

#%%
# Snapchat basado realidad arguementada adjuntando  gafas

glasses = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/glasses.png',-1)
face_test = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/face_test.png')

face_gray = cv2.cvtColor(face_test,cv2.COLOR_BGR2GRAY)
glasses_mask = glasses[:,:,3]
glasses = glasses[:,:,0:3]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyepair_cascade = cv2.CascadeClassifier("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/parojos.xml")

faces = face_cascade.detectMultiScale(face_gray,1.3,5)

for (x,y,w,h) in faces:
    roi_gray = face_gray[y:y+h,x:x+w]
    roi_color = face_test[y:y+h,x:x+w]
    eyespair = eyepair_cascade.detectMultiScale(roi_gray)
    
    for (nx,ny,nw,nh) in eyespair:
        x1 = int(nx-nw/10)
        x2 = int((nx+nw)+nw/10)
        y1 = int(ny)
        y2 = int(ny+nh+nh/2)
        
        if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
            continue
        
        img_glasses_ret_width = int(x2-x1)
        img_blasses_ret_height = int(y2-y1)
        mask = cv2.resize(glasses_mask,(img_glasses_ret_width,img_blasses_ret_height))
        mask_inv = cv2.bitwise_not(mask)
        img = cv2.resize(glasses,(img_glasses_ret_width,img_blasses_ret_height))
        roi = roi_color[y1:y2,x1:x2]
        roi_background = cv2.bitwise_and(roi,roi,mask=mask_inv)
        roi_foreground = cv2.bitwise_and(img,img,mask=mask)
        
        res = cv2.add(roi_background,roi_foreground)
        roi_color[y1:y2,x1:x2]=res

plt.imshow(face_test[:,:,::-1])
plt.title('Snapchat-based glasses')
plt.axis('off')
plt.show()
        

#%%
# Deteccion de  codigo QR
"""
Para detectar codigos QR opencv ofrece una funcion cv2.detectAndDecode() esta funcion retorna:
    Un array de vertices del codigo Qr encontrado. Este array puede ser basio si el QR  no se encontro.
    El rectificado y binarizado codico QR es retornado.
    El dato asociado al codigo QR es retornado.
    
"""
#%%

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/qrcode_rotate_45_image.png')
qr_detector = cv2.QRCodeDetector()
image_copy = image.copy()
data,bbox,rectified_qr_code = qr_detector.detectAndDecode(image)

def show_qr_detector(image,pts):
    pts = np.int32(pts).reshape(-1,2)
    
    for j in range(pts.shape[0]):
        cv2.line(image,tuple(pts[j]),tuple(pts[(j+1)%pts.shape[0]]),(255,0,0),5)
    
    for j in range(pts.shape[0]):
        cv2.circle(image,tuple(pts[j]),10,(255,0,255),-1)



if len(data) > 0:
    print("Decoded Data: {}".format(data))
    show_qr_detector(image_copy,bbox)

        
fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('silver')
plt.suptitle('QR code  detection',fontsize=12,fontweight='bold')
plt.subplot(121)
plt.imshow(image[:,:,::-1])
plt.title('Code QR')
plt.axis('off')
plt.subplot(122)
plt.imshow(image_copy[:,:,::-1])
plt.title('Decode QR')
plt.axis('off')
plt.show()
#%%

