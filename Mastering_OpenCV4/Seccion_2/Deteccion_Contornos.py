# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:39:24 2021

@author: MBI
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = (10,8)
#%%
# Deteccion de contornos

def get_one_contour():
    "Returns a fixed cotour"
    cnts = [np.array(
            [[[600, 320]], [[563, 460]], [[460, 562]], [[320, 600]], [[180, 563]],
            [[78, 460]], [[40, 320]], [[77, 180]], [[179, 78]], [[319, 40]], [[459,
            77]], [[562, 179]]], dtype=np.int32)]
    return cnts

contour = get_one_contour()
print("[*] Detected contours: '{}' ".format(len(contour)))
print("Contour shape: '{}' ".format(contour[0].shape))
#%%
def draw_contour_points(img,cnts,color):
    for cnt in cnts:
        squeeze = np.squeeze(cnt)
        for p in squeeze:
            p = array_to_tuple(p)
            cv2.circle(img,p,10,color,-1)
    return img

def array_to_tuple(arr):
    return tuple(arr.reshape(1,-1)[0])

def draw_contour_line(img,cnts,color):
    pts = []
    for cnt in cnts:
        squeeze = np.squeeze(cnt)
        for p in squeeze:
            p = array_to_tuple(p)
            pts.append(p)
    
    for pt in range(len(pts)):
        cv2.line(img,pts[len(pts)-(len(pts)-pt)],pts[pt+1],color,3,lineType=cv2.LINE_AA)
        if pt == (len(pts)-2):
            break
    cv2.line(img,pts[len(pts)-1],pts[0],color,3,lineType=cv2.LINE_AA)
    return img

def draw_contour_line_circle(img,cnts,color_c,color_l):
    image = draw_contour_points(img,cnts,color_c)
    image = draw_contour_line(image,cnts,color_l)
    return image
    
img1 = np.zeros(shape=(800,800,3),dtype='uint8')
img2 = np.zeros(shape=(800,800,3),dtype='uint8')
img3 = np.zeros(shape=(800,800,3),dtype='uint8')
img_c = draw_contour_points(img1,contour,(0,0,255))
img_l = draw_contour_line(img2,contour,(255,0,0))
img_cl = draw_contour_line_circle(img3,contour,(0,0,255),(255,0,0))

plt.subplot(131)
plt.imshow(img_c)
plt.title('Contours points')
plt.axis('off')
plt.subplot(132)
plt.imshow(img_l)
plt.title('Contour lines')
plt.axis('off')
plt.subplot(133)
plt.imshow(img_cl)
plt.title('Contour lines/points')
plt.axis('off')
plt.show()

#%%
def build_sample_image():
    img = np.ones(shape=(500,500,3),dtype='uint8') * 70
    cv2.rectangle(img,(100,100),(300,300),(255,0,255),-1)
    cv2.circle(img,(400,400),100,(255,255,0),-1)
    return img

plt.imshow(build_sample_image())
plt.title('Sambple image')
plt.axis('off')
plt.show()
#%%
def build_sample_image2():
    img = np.ones(shape=(500,500,3),dtype='uint8') * 70
    cv2.rectangle(img,(100,100),(300,300),(255,0,255),-1)
    cv2.rectangle(img,(150,150),(250,250),(70,70,70),-1)
    cv2.circle(img,(400,400),100,(255,255,0),-1)
    cv2.circle(img,(400,400),50,(70,70,70),-1)
    return img

plt.imshow(build_sample_image2())
plt.title('Contours')
plt.axis('off')
plt.show()
    
#%%
"""
Funcion usada para detectar contornos cv2.findContours().Detecta contornos en una imagen
binaria.
Parametros:
    sr : fuente de imagen
    mode : cv2.RETR_EXTERNAL (salida solamente externa de contornos)
           cv2.RETR_LIST (salida de todos los contornos sin relacion de gerarquia)
           cv2.RETR_TREE (salida de todos los contornos por una relacion de gerarquia)
    salida : vector de relacion de gerarquia que contiene la informacion de relacion de 
             cada contorno[i],hierarchy[i][j] con j en el rango de [0,3]
    
    hierarchy[i][0]: indice del contorno  siguiente al mismo nivel de gerarquia
    hierarchy[i][1]: indice del contorno previo al mismo nivel de gerarquia
    hierarchy[i][2]: indice del primer contorno hijo
    hierarchy[i][3]: indice del contorno padre
    
    method : parametro que establece el metodo de aproximacion usado para sacar los puntos 
             que consiernen a cada contorno
"""
def draw_contorno(img,cnts,color,tl):
    for cn in cnts:
        cv2.drawContours(img,[cn],0,color,thickness=tl)
    return img
        
image = build_sample_image2()

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

ret,threshold = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)

contour1,h1 = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
contour2,h2 = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
contour3,h3 = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

gray_image1 = gray_image.copy()
gray_image2 = gray_image.copy()
gray_image3 = gray_image.copy()

image_cont1 = draw_contorno(gray_image1,contour1,(255,0,255),3)
image_cont2 = draw_contorno(gray_image2,contour2,(255,0,0),3)
image_cont3 = draw_contorno(gray_image3, contour3,(255,0,255),3)

plt.subplot(151)
plt.imshow(image)
plt.title('Image')
plt.axis('off')
plt.subplot(152)
plt.imshow(threshold)
plt.title('Threshodd 100')
plt.axis('off')
plt.subplot(153)
plt.imshow(image_cont1)
plt.title('Contours RETR_EXTERNAL')
plt.axis('off')
plt.subplot(154)
plt.imshow(image_cont2)
plt.title('Contours RETR_LIST')
plt.axis('off')
plt.subplot(155)
plt.imshow(image_cont3)
plt.title('Contours RETR_TREE')
plt.axis('off')
plt.show()

#%%
# Compresion de contornos
"""
Tipos de metodos de compresion:
    cv2.CHAIN_APPROX_NONE:No se elige compresion alguna
    cv2.CHAIN_APPROX_SIMPLE:Puede ser usado para comprimir los contornos detectados
                            comprime horizontal,vertical y diagonal segmentos de contornos
                            preservando solamente los puntos terminales.
    cv2.CHAIN_APPROX_TC89_L1
    cv2.CHAIN_APPROX_TC89_KCOS

Nota: Los 2 ultimos metodos son metodos no parametricos.El primer paso del algoritmo determina
la region de soporte (ROS) para cada uno de los puntos basados sobre sus propiedades locales.
Seguido computa la medida relativa de significancia  de cada uno  de los puntos,finalmente los do-
mineos de puntos son detectados por un proceso de no significancia , correspondiente a los diferentes
grados de exactitud de la medidas de la curva discreta.
                            
"""
# Mostrando todos los metodos

contour4,h4 = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contour5,h5 = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
contour6,h6 = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_KCOS)

gray_image4 = gray_image.copy()
gray_image5 = gray_image.copy()
gray_image6 = gray_image.copy()

image_cont4 = draw_contorno(gray_image4,contour4,(255,0,255),2)
image_cont5 = draw_contorno(gray_image5,contour5,(255,0,255),2)
image_cont6 = draw_contorno(gray_image6,contour6,(255,0,255),2)  

plt.subplot(231)
plt.imshow(image)
plt.title('Image')
plt.axis('off')
plt.subplot(232)
plt.imshow(threshold)
plt.title('Threshodd 100')
plt.axis('off')
plt.subplot(233)
plt.imshow(image_cont2)
plt.title('NONE')
plt.axis('off')
plt.subplot(234)
plt.imshow(image_cont4)
plt.title('Simple')
plt.axis('off')
plt.subplot(235)
plt.imshow(image_cont5)
plt.title('TC89_L1')
plt.axis('off')
plt.subplot(236)
plt.imshow(image_cont6)
plt.title('TC89_KCOS')
plt.axis('off')
plt.show()

#%%




