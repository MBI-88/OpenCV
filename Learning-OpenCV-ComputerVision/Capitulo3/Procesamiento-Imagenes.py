# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:13:17 2021

@author: MBI
"""
import cv2 
import  numpy as np 
from scipy import ndimage

#%% HPFS y LPFS

kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, -1, -1, -1, -1]])

img = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo3/Learning OpenCV 4 Computer Vision with Python 3_page364_image31.jpg",0)

k3 = ndimage.convolve(img,kernel_3x3)
k5 = ndimage.convolve(img,kernel_5x5)

blurred = cv2.GaussianBlur(img,(17,17),0)
g_hpf = img - blurred

cv2.imshow("3x3",k3)
cv2.imshow("5x5",k5)
cv2.imshow("blurred",blurred)
cv2.imshow("g_hpf",g_hpf)

while  True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cv2.destroyAllWindows()
#%% Definicon de filtros
"""
medianBlur : Es efectivo para remover ruido digital,especialmente en imagenes a colores.

Laplacian : Para deteccion de bordes el cual produce lineas de bordes oscuros, especial en imagenes en grises.

Nota: El orden de aplicar los filtros  es primero un medianBlur, llevar la imagen a grises y aplicar un Laplacian despues volver la imagen a colores.

"""

"""
Entendiendo el uso de kernel:
    
    kernel = numpy.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    
A partir del kernel de arriba tomado como ejemplo. El pixel de interes tiene un peso de 9 y es becino inmediato de cada uno de los pixel con peso -1. Para  el pixel de interes, el color de salida sera 9 veces el color de entrada, 
menos  los colores de entrada de los 8 pixeles adyacentes. Si el pixel de interes ya tiene una pequeña diferencia respecto a sus vecinos, 
esta diferencia se intentificara. El efecto es que parecera una imagen mas nitidez en contraste con el incremento de sus vecinos.
"""
#%% Deteccion de bordes con Canny

img = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo3/Learning OpenCV 4 Computer Vision with Python 3_page364_image31.jpg",0)

cv2.imwrite("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo3/canny.jpg",cv2.Canny(img,200,300,(4,4)))
cv2.imshow("canny",cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo3/canny.jpg"))

while True:
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyWindow("canny")

#%%  Deteccion de contornos

img = np.zeros((200,200),dtype='uint8')
img[50:150,50:150] = 255

ret,thresh = cv2.threshold(img,127,255,0)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color,contours,-1,(0,255,0),2)

cv2.imshow("Contornos",color)
 
while True:
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyWindow("Contornos")
#%%  Bondaridades de caja, area minima de rectangulo y minimo circulo cerrado.

img = cv2.pyrDown(cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo3/hammer.jpg",cv2.IMREAD_UNCHANGED)) # Hace un bajo muestreo de la imagen de entrada.

img_ = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img_,127,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),2)
    (X,Y),radius = cv2.minEnclosingCircle(c)
    center = (int(X),int(Y))
    radius = int(radius)
    img = cv2.circle(img,center,radius,(0,255,0),2)
    
cv2.drawContours(img,contours,-1,(255,0,0),2)
cv2.imshow("Contours",img)

while True:
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyWindow("Contours")
#%%
# Contornos conversos y algoritmo Douglas-Peucker

"""
Definicion y uso de la funcion cv2.approxPolyDP:
    Esta funcion tiene tres parametros: 
        .un contorno
        .un valor epsilon representando la maxima discrepancia entre el contorno original y la aproximacion (el valor mas bajo, da una aproximacion mas cerrada)
        .una bandera booleano para especificar si es cerrado (True) o abierto (False)

Nota: OpenCv provee otra funcion para el calculo de los contornos cv2.convexHull() la cual se usa para obtener informacion de contornos de formas convexas.
"""

img = cv2.pyrDown(cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo3/hammer.jpg",cv2.IMREAD_UNCHANGED))

ret,thresh = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)
contour,her = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

black = np.zeros_like(img)
for cnt in contour:
    epsilon = 0.01 * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    hull = cv2.convexHull(cnt)
    cv2.drawContours(black,[cnt],-1,(0,255,0),2)
    cv2.drawContours(black,[approx],-1,(255,255,0),2)
    cv2.drawContours(black,[hull],-1,(0,0,255),2)
    
cv2.imshow("Hull",black)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Hull")

#%%
# Detectando lineas

"""
Parmetros de la funcion HoughLinesp():
    .la imagen
    .la resolucion o tamaño del paso para usar cuando se encuentren lineas, rho es el tamaño del paso en pixeles, theta es el tamaño del paso rotacional en radianes.
    .threshold, el cual representa el umbral por debajo del cual se descarta la linea.
    .minLineLength y maxLineGap: valor extremos .
"""

img = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo3/lines.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,120)
minLineLenght = 20
maxLineGap = 5 
lines = cv2.HoughLinesP(edges,1,np.pi/180.0,20,minLineLenght,maxLineGap)

for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


cv2.imshow("Edges",edges)
cv2.imshow("lines",img)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



#%%
# Detectando circulos

planets = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo3/planet_glow.jpg")

gray_img = cv2.cvtColor(planets,cv2.COLOR_BGR2GRAY)
gray_img = cv2.medianBlur(gray_img,5)

circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,120,param1=100,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv2.circle(planets,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(planets,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow("HoughCircles",planets)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("HoughCircles")
#%%



