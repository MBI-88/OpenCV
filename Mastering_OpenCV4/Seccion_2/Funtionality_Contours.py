# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:03:24 2021

@author: MBI
"""
import cv2 
import numpy as np
import matplotlib.pyplot as plt

#%%
# Mas funcionalidadaes relacionada con contornos

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/shape_features.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

image_extremepoint = image_gray.copy()
image_area  = image_gray.copy()
image_elip = image_gray.copy()
image_rect = image_gray.copy()
image_mincircle = image_gray.copy()
image_polydp = image_gray.copy()

ret,thresh = cv2.threshold(image_gray,50,255,cv2.THRESH_BINARY)
contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


def extreme_point(contours):
    index_min_x = tuple(contours[contours[:,:,0].argmin()][0])
    index_min_y = tuple(contours[contours[:,:,1].argmin()][0])
    index_max_x = tuple(contours[contours[:,:,0].argmax()][0])
    index_max_y = tuple(contours[contours[:,:,1].argmax()][0])
    
    return index_min_x,index_max_x,index_min_y,index_max_y

def draw_ext_point(img,coord_poits):
    for i in coord_poits:
        cv2.circle(img,i,20,(0,255,0),-1)
    return img

def draw_boundingRect(img,contours):
    x,y,w,h = cv2.boundingRect(contours[0])
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),5)
    return img  

def draw_rotatedRect(img,contours):
    rotated_rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)
    cv2.polylines(img,[box],True,(255,0,255),5)
    return img

def draw_enclosingCircle(img,contours):
    (a,b),radius = cv2.minEnclosingCircle(contours[0])
    cv2.circle(img,(int(a),int(b)),int(radius),(255,0,180),5)
    return img

def draw_elipse(img,contours):
    elipse = cv2.fitEllipse(contours[0])
    cv2.ellipse(img,elipse,(0,255,255),10)
    return img 

def draw_approxPoly(img,contours):
    epsilon = 0.01 * cv2.arcLength(contours[0],True)
    approx = cv2.approxPolyDP(contours[0],epsilon,True)
    for i in [approx]:
        cv2.drawContours(img,[i],0,(0,255,255),15)
    
    for i in [approx]:
        squeeze = np.squeeze(i)
        
    for p in squeeze:
        pp = tuple(p.reshape(1,-1)[0])
        cv2.circle(img,pp,15,(255,0,255),-1)
    return img
    
fig = plt.figure(figsize=(13, 9))
plt.suptitle("Functionality related to contours", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

plt.subplot(321)
plt.imshow(draw_ext_point(image_extremepoint,extreme_point(contours[0])))
plt.title('Image and extreme points')
plt.axis('off')
plt.subplot(322)
plt.imshow(draw_boundingRect(image_area,contours))
plt.title('cv2.boundingRect()')
plt.axis('off')
plt.subplot(323)
plt.imshow(draw_rotatedRect(image_rect,contours))
plt.title('cv2.minAreaRect()')
plt.axis('off')
plt.subplot(324)
plt.imshow(draw_enclosingCircle(image_mincircle,contours))
plt.title('cv2.minEnclosingCircle()')
plt.axis('off')
plt.subplot(325)
plt.imshow(draw_elipse(image_elip,contours))
plt.title('cv2.ellipse()')
plt.axis('off')
plt.subplot(326)
plt.imshow(draw_approxPoly(image_polydp,contours))
plt.title('cv2.approxPolyDP()')
plt.axis('off')
plt.show()

#%%
