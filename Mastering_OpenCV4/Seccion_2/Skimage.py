# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 10:01:24 2021

@author: MBI
"""

import skimage as img
import  cv2
import  matplotlib.pyplot as plt

"Skimage es una libreria de procesado de  imagenes"
#%%
# Umbralizacion en skimage

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/leaf.png')
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

threshold = img.filters.threshold_otsu(gray_image)
binary = gray_image > threshold
binary = img.img_as_ubyte(binary)

histograma = cv2.calcHist([binary],[0],None,[256],[0,256])

plt.subplot(221)
plt.imshow(image[:,:,::-1])
plt.title('Image')
plt.axis('off')
plt.subplot(222)
plt.imshow(gray_image)
plt.title('Gray Image')
plt.axis('off')
plt.subplot(223)
plt.plot(histograma,color='r')
plt.axvline(x=threshold,color='m',linestyle='--')
plt.title('Histogram')
plt.subplot(224)
plt.imshow(binary)
plt.title('Image thresh skimage')
plt.axis('off')
plt.show()

#%%
# Intentando mas tecnicas con skimage. Comparando algoritmos de skimage

sudoku = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/sudoku.png')
gray_sudoku = cv2.cvtColor(sudoku,cv2.COLOR_BGR2GRAY)


thresh_ostu = img.filters.threshold_otsu(gray_sudoku)
binary_otsu = gray_sudoku > thresh_ostu
binary_otsu = img.img_as_ubyte(binary_otsu)

thresh_niblack = img.filters.threshold_niblack(gray_sudoku,window_size=25,k=0.8)
binary_niblack = gray_sudoku > thresh_niblack
binary_niblack = img.img_as_ubyte(binary_niblack)

thresh_sauvola = img.filters.threshold_sauvola(gray_sudoku,window_size=25)
binary_sauvola = gray_sudoku > thresh_sauvola
binary_sauvola = img.img_as_ubyte(binary_sauvola)

thresh_triangle = img.filters.threshold_triangle(gray_sudoku)
binary_triangle = gray_sudoku > thresh_triangle
binary_triangle = img.img_as_ubyte(binary_triangle)

plt.subplot(231)
plt.imshow(sudoku[:,:,::-1])
plt.title('Sudoku')
plt.axis('off')
plt.subplot(232)
plt.imshow(gray_sudoku)
plt.title('Gray sudoku')
plt.axis('off')
plt.subplot(233)
plt.imshow(binary_triangle)
plt.title('Triangle')
plt.axis('off')
plt.subplot(234)
plt.imshow(binary_niblack)
plt.title('Niblack')
plt.axis('off')
plt.subplot(235)
plt.imshow(binary_otsu)
plt.title('Otsu')
plt.axis('off')
plt.subplot(236)
plt.imshow(binary_sauvola)
plt.title('Saubola')
plt.axis('off')
plt.show()

#%%







#%%