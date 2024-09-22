#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,15)


# In[2]:


# Mescla de imagenes usando Sobel

image=cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/dog-01.jpg')
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gradiente_x=cv2.Sobel(image_gray,cv2.CV_16S,1,0,3)
gradiente_y=cv2.Sobel(image_gray,cv2.CV_16S,0,1,3)
abs_gradient_x=cv2.convertScaleAbs(gradiente_x)
abso_gradient_y=cv2.convertScaleAbs(gradiente_y)

sobel_image=cv2.addWeighted(abs_gradient_x,0.5,abso_gradient_y,0.5,0)
plt.subplot(221)
plt.imshow(image)
plt.title('Image')
plt.axis('off')
plt.subplot(222)
plt.imshow(sobel_image)
plt.title('Sobel Image')
plt.axis('off')
plt.subplot(223)
plt.imshow(gradiente_x)
plt.title('Gradiente X')
plt.axis('off')
plt.subplot(224)
plt.imshow(gradiente_y)
plt.title('Gradiente Y')
plt.axis('off')
plt.show()


# In[4]:


image_1=cv2.imread('opencv_binary_logo_250.png')
image_2=cv2.imread('lenna_250.png')
image_rgb=image_2[:,:,::-1]
# La  operacion se hace con una imagen binaria y las dos de la misma forma
# Nota: Si la imagen no es en formato de grices se debe cambiar a GBR
bit_and=cv2.bitwise_and(image_rgb,image_1)
bit_or=cv2.bitwise_or(image_rgb,image_1)
bit_not=cv2.bitwise_not(image_rgb,image_1)
bit_xor=cv2.bitwise_xor(image_rgb,image_1)

plt.subplot(221)
plt.imshow(bit_and)
plt.title('Bitwise_and')
plt.axis('off')
plt.subplot(222)
plt.imshow(bit_or)
plt.title('Bitwise_or')
plt.axis('off')
plt.subplot(223)
plt.imshow(bit_not)
plt.title('Bitwise not')
plt.axis('off')
plt.subplot(224)
plt.imshow(bit_xor)
plt.title('Bitwise xor')
plt.axis('off')
plt.show()


# In[ ]:




