# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:01:53 2021

@author: MBI
"""
import cv2
import numpy as np
#%%

"""
Definicion de la funcion: cv2.cornerHarris().

Esta funcion captura bordes de una imagen en grices, el parametro mas importante es el 3ro que es la apertura o tmaño del kernel Sobel este valor esta entre  3-31. Un valor bajo (alta precision) toda  las lineas diagonales seran registradas como esquinas.Un valor alto (baja precision) solo las esquinas de cada cuadarado seran detectadas. Esta funcion devuelve una imagen en float cada valor de esta imagen es un puntaje correspondiente a cada pixel de la imagen fuente. Un puntaje alto indica que el pixel se parece mas a una esquina.
"""

img = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo6/chess_board.png")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray,2,23,0.04)
img[dst > 0.01 *  dst.max()] = [0,0,255]
cv2.imshow("Corner",img)
while (cv2.waitKey(1) == -1):
    continue
cv2.destroyWindow("Corner")

#%%
# Deteccion de variables DoG y extraccion de descriptores  SIFT. Este algoritmp es invariante al escalado de imagenes.

car = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo6/cars_small.jpg")

gray = cv2.cvtColor(car,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints,descriptors = sift.detectAndCompute(gray,None)

cv2.drawKeypoints(car,keypoints,car,(51,163,236),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Corner",car)
while (cv2.waitKey(1) == -1):
    continue
cv2.destroyWindow("Corner")
"""
Anatomia de los keypoints:
    Cada uno de los puntos es una instancia de la clase cv2.KeyPoint la cual tiene las siguientes propiedades: 
        . pt (point) propiedad que contiene las coordenadas x,y de los puntos en la imagen.
        . size propiedad  que indica el diametro de la variable.
        . angle propiedad que indica la orientacion de una variable, se dan en radianes.
        . response propiedad que indica la fuerza de el punto.Algunas variables son clasificadas por SIFT como mas fuertes que otras.
        . octave propiedad que indica la capa en la piramide de imagen donde la variable fue encontrada.
        . class_id propiedad que puede ser  usada para  asignar un identificador personalizado  a un punto o gurpo de puntos.
"""
#%%
# Deteccion Hessian rapida de variables y extraccion SURF de descriptores.

car = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo6/cars_small.jpg")

gray = cv2.cvtColor(car,cv2.COLOR_BGR2GRAY)

surf  = cv2.xfeatures2d.SURF_create(8000)
keypoints,descriptors = surf.detectAndCompute(gray,None)

cv2.drawKeypoints(car,keypoints,car,(51,163,236),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Corner",car)
while (cv2.waitKey(1) == -1):
    continue
cv2.destroyWindow("Corner")

# Nota: El algoritmo no trabaja por que hay que abilitarlo en una compilacion desde cero.

#%%
# Ejecutando un logo en 2 imagenes.

img_0 = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo6/Learning OpenCV 4 Computer Vision with Python 3_page364_image79.jpg")

img_1 = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo6/Learning OpenCV 4 Computer Vision with Python 3_page364_image78.jpg")

orb = cv2.ORB_create()
kp0,des0 = orb.detectAndCompute(img_0,None)
kp1,des1 = orb.detectAndCompute(img_1,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches = bf.match(des0,des1)

matches = sorted(matches,key=lambda x: x.distance)

img_matches = cv2.drawMatches(img_0,kp0,img_1,kp1,matches[:25],img_1,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Matches",img_matches)

while (cv2.waitKey(1) == -1):
    continue
cv2.destroyWindow("Matches")

#%%
# Filtando ejecutores usando K-Nearest.

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

pairs_of_matches = bf.knnMatch(des0,des1,k=2)
pairs_of_matches = sorted(pairs_of_matches, key=lambda x: x[0].distance)

img_pairs_of_matckes = cv2.drawMatchesKnn(img_0,kp0,img_1,kp1,pairs_of_matches[:25],img_1,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Matches_filter",img_pairs_of_matckes)

while (cv2.waitKey(1) == -1):
    continue
cv2.destroyWindow("Matches_filter")


#%%
# Aplicando la prueba del radio.

matches = [x[0] for x in pairs_of_matches 
           if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]

img_matckes = cv2.drawMatches(img_0,kp0,img_1,kp1,matches[:25],img_1,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Matches test ratio",img_matckes)

while (cv2.waitKey(1) == -1):
    continue
cv2.destroyWindow("Matches test ratio")

#%%
# Ejecutores usando FLANN
# Nota: La documentacion de Flann recomienda entre 1 y 16 arboles

img0 = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo6/Learning OpenCV 4 Computer Vision with Python 3_page364_image84.jpg",cv2.IMREAD_GRAYSCALE)

img1 = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo6/Learning OpenCV 4 Computer Vision with Python 3_page364_image85.jpg",cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
kp0,des0 = sift.detectAndCompute(img0,None)
kp1,des1 = sift.detectAndCompute(img1,None)

flann_index_sktree = 1
index_params = dict(algorithm=flann_index_sktree,trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des0,des1,k=2)

mask_matches = [[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        mask_matches[i] = [1,0]

img_matches = cv2.drawMatchesKnn(img0,kp0,img1,kp1,matches,None,matchColor=(0,255,0),singlePointColor=(0,0,255),matchesMask=mask_matches,flags=0)

cv2.imshow("Matches test ratio",img_matches)

while (cv2.waitKey(1) == -1):
    continue
cv2.destroyWindow("Matches test ratio")
#%%
# Eligiendo homografia con Flann

image_0 = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo6/Learning OpenCV 4 Computer Vision with Python 3_page364_image88.jpg",cv2.IMREAD_GRAYSCALE)
image_1 = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo6/Learning OpenCV 4 Computer Vision with Python 3_page364_image87.jpg",cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
kp0,des0 = sift.detectAndCompute(image_0,None)
kp1,des1 = sift.detectAndCompute(image_1,None)

flann_index_sktree = 1
index_params = dict(algorithm=flann_index_sktree,trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des0,des1,k=2)

good_matches = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

min_num_good_matches = 10

if len(good_matches) >= min_num_good_matches:
    src_pts = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
    mask_matches = mask.ravel().tolist()

    h,w = image_0.shape
    src_corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst_corners = cv2.perspectiveTransform(src_corners,M)
    dst_corners = dst_corners.astype(np.int32)

    num_corners = len(dst_corners)
    for i in range(num_corners):
        x0,y0 = dst_corners[i][0]
        if i == num_corners - 1:
            next_i = 0
    
        else:
            next_i = i + 1
        x1,y1 = dst_corners[next_i][0]
        cv2.line(image_1,(x0,y0),(x1,y1),255,3,cv2.LINE_AA)

    image_matches = cv2.drawMatches(image_0,kp0,image_1,kp1,good_matches,None,matchesMask=mask_matches,flags=2)

cv2.imshow("Matches test",image_matches)

while (cv2.waitKey(1) == -1):
    continue
cv2.destroyWindow("Matches test")


#%%




