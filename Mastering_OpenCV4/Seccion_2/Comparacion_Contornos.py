# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:58:35 2021

@author: MBI
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
#%%
"""
El momnetum invariante Hu puede ser usado para ejecutar un objeto y reconocimento. OpenCv
provee cv2.matchShapes(),la cual  puede ser  usada para comparar 2 contornos usando 3 metodos
de comparacion. Los metodos son: cv2.CONTOURS_MATCH_I1,cv2.CONTOURS_MATCH_I2,cv2.CONTOURS_MATCH_I3

Si denotamos un  objeto A como primer objeto y B como segundo objeto entonces se aplica lo siguiente

mi^A = sign(hi^A) * log hi^A
mi^B = sign(hi^B) * log hi^A

donde hi^A y hi^B son los momentum Hu de A y B respectivamente

cv2.CONTOURS_MATCH_I1 : I1(A,B) = Sumatoriai...7 |(1/mi^A - 1/mi^B)|

cv2.CONTOURS_MATCH_I2 : I2(A,B) = Sumatoriai...7 |(mi^A - mi^B)|

cv2.CONTOURS_MATCH_I3 : I3(A,B) = Sumatoriai...7 |(mi^A - mi^B)|
"""

def perfect_circle():
    image_ref = np.zeros(shape=(500,500,3),dtype='uint8')
    cv2.circle(image_ref,(250,250),200,(255,255,255),1)
    return image_ref

def get_ceintroid(contours):
    M = cv2.moments(contours)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx,cy

def get_text_position(text,pt,font_size,font_scale,thickness):
    text = cv2.getTextSize(text,font_size,font_scale,thickness)[0]
    text_x = pt[0] - text[0]/2
    text_y = pt[1] + text[1]/2
    return round(text_x),round(text_y)    


image_macth = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/match_shapes.png')
image_ref = perfect_circle()

image_I1 = image_macth.copy()
image_I2 = image_macth.copy()
image_I3 = image_macth.copy()

gray_macth = cv2.cvtColor(image_macth,cv2.COLOR_BGR2GRAY)
gray_ref = cv2.cvtColor(image_ref,cv2.COLOR_BGR2GRAY)

re1,thres1 = cv2.threshold(gray_macth,70,255,cv2.THRESH_BINARY_INV)
re2,thres2 = cv2.threshold(gray_ref,70,255,cv2.THRESH_BINARY)

contours_ref,heiar_ref = cv2.findContours(thres2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
contours_macth,heiar_macth = cv2.findContours(thres1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

print('Detected contours macth {} detected contour ref {}'.format(len(contours_macth),len(contours_ref)))

for cnts in contours_macth:
    cX,cY = get_ceintroid(cnts)
    ret1 = cv2.matchShapes(contours_ref[0],cnts,cv2.CONTOURS_MATCH_I1,0.0)
    ret2 = cv2.matchShapes(contours_ref[0],cnts,cv2.CONTOURS_MATCH_I2,0.0)
    ret3 = cv2.matchShapes(contours_ref[0],cnts,cv2.CONTOURS_MATCH_I3,0.0)
    
    (x_1,y_1) = get_text_position(str(round(ret1,3)),(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,1.2,3)
    (x_2,y_2) = get_text_position(str(round(ret2,3)),(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,1.2,3)
    (x_3,y_3) = get_text_position(str(round(ret3,3)),(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,1.2,3)
    
    cv2.putText(image_I1,str(round(ret1,3)),(x_1,y_1),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),3)
    cv2.putText(image_I2,str(round(ret2,3)),(x_2,y_2),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
    cv2.putText(image_I3,str(round(ret3,3)),(x_3,y_3),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

fig = plt.figure(figsize=(18,6))
plt.suptitle('Maching contours')
fig.patch.set_facecolor('silver')
plt.subplot(131)
plt.imshow(image_I1[:,:,::-1])
plt.title('Matching scores (method I1)')
plt.axis('off')
plt.subplot(132)
plt.imshow(image_I2[:,:,::-1])
plt.title('Matching scores (method I2)')
plt.axis('off')
plt.subplot(133)
plt.imshow(image_I3[:,:,::-1])
plt.title('Matching scores (method I3)')
plt.axis('off')
plt.show()




#%%

