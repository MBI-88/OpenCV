# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:01:08 2021

@author: MBI
"""
import cv2 
import matplotlib.pyplot as plt
import numpy as np

#%%
image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/shapes.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_1 = image.copy()
image_2 = image.copy()

ret,threshold = cv2.threshold(image_gray,50,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

dic_fig = {0:'square',1:'circle',5:'pentagon',6:'hexagon',3:'triangle',4:'rectangle'}

def detect_contour(img,contours):
    epsilon = 0.004 * cv2.arcLength(contours,True) 
    approx = cv2.approxPolyDP(contours,epsilon,True)
    
    if len(approx) == 3:
        contour_shape = dic_fig[3]
    elif len(approx) == 6:
        contour_shape = dic_fig[6]
    elif len(approx) == 5:
        contour_shape = dic_fig[5]
    elif len(approx) == 4:
        x,y,w,h = cv2.boundingRect(approx)
        aspect_ratio = float(w)/h
        if 0.90 < aspect_ratio < 1.10:
            contour_shape = dic_fig[0]
        else:
            contour_shape = dic_fig[4]
    else:
        contour_shape = dic_fig[1]
        
        
    for i in [approx]:
        cv2.drawContours(img,[i],0,(255,255,255),3)
        squeeze = np.squeeze(i)
    for p in squeeze:
        pp = tuple(p.reshape(1,-1)[0])
        cv2.circle(img,pp,10,(255,255,255),-1)
        
    return contour_shape

def get_centroi(contours):
    M = cv2.moments(contours)
    X = int(M['m10']/M['m00'])
    Y = int(M['m01']/M['m00'])
    return X,Y

def get_position_text(text,pt,font_size,font_scale,thickness):
    text_size = cv2.getTextSize(text,font_size,font_scale, thickness)[0]
    text_x = pt[0] - text_size[0]/2
    text_y = pt[1] + text_size[1]/2
    return round(text_x),round(text_y)

print('Detected contours {}'.format(len(contours)))
for cn in contours:
    nameshape = detect_contour(image_1,cn)
    cX,xY = get_centroi(cn)
    X,Y = get_position_text(nameshape,(cX,xY),cv2.FONT_HERSHEY_SIMPLEX,1.6,3)
    cv2.putText(image_2,nameshape,(X,Y),cv2.FONT_HERSHEY_SIMPLEX,1.6,(255,255,255),3)
    
    
fig = plt.figure(figsize=(12,9))
plt.suptitle('Shape recognition based cv2.approxPolyDP()')
fig.patch.set_facecolor('silver')
plt.subplot(221)
plt.imshow(image[:,:,::-1])
plt.title('Image')
plt.axis('off')
plt.subplot(222)
plt.imshow(threshold)
plt.title('Threshold = 50')
plt.axis('off')
plt.subplot(223)
plt.imshow(image_1[:,:,::-1])
plt.title('Contours outline ')
plt.axis('off')
plt.subplot(224)
plt.imshow(image_2[:,:,::-1])
plt.title('Contours recognition')
plt.axis('off')
plt.show()
#%%