# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:19:59 2021

@author: MBI
"""

import cv2
import matplotlib.pyplot as plt
#%%
# Filtrando contornos
# Ejemplo de uso de zip()

coordinate = ['x','y','z']
value = [5,4,3]
result = zip(coordinate,value)
print(list(result))
print('\n')
c,v = zip(*zip(coordinate,value))
print('c = ',c)
print('v = ',v)
print('\n')

# Incorporando sorted()
print(sorted(zip(value,coordinate)))
print('\n')
c,v = zip(*sorted(zip(value,coordinate)))
print('c = ',c)
print('v = ',v)
#%%
def sort_contours_size(cnts):
    cnts_sizes = [cv2.contourArea(contour) for contour in cnts]
    (cnts_sizes,cnts) = zip(*sorted(zip(cnts_sizes,cnts)))
    return cnts_sizes,cnts


def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """Gives the coordinates to draw centered"""

    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)

def get_centroi(contour):
    moments = cv2.moments(contour)
    X = round(moments['m10']/moments['m00'])
    Y = round(moments['m01']/moments['m00'])
    return X,Y

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/shapes_sizes.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret,threshold = cv2.threshold(gray_image,50,255,cv2.THRESH_BINARY)
image_copy = image.copy()

contours,hier = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

area,cnts = sort_contours_size(contours)
for i,valor in enumerate(cnts,start=1):
    X,Y = get_centroi(valor)
    text_draw = str(i)
    w,h = get_position_to_draw(text_draw,(X,Y),cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    cv2.putText(image_copy,text_draw,(w,h),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

fig = plt.figure(figsize=(9, 9))
plt.suptitle("Sort contours by size", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')
plt.subplot(211)
plt.imshow(image[:,:,::-1])
plt.title('Image')
plt.axis('off')
plt.subplot(212)
plt.imshow(image_copy[:,:,::-1])
plt.title('Result')
plt.axis('off')
plt.show()
#%%


