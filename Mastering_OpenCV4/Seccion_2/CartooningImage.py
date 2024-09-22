#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,15)


# In[2]:


image=cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/cat-01.jpg')
image=image[:,:,::-1]

pencil_image_0,pencil_image_1=cv2.pencilSketch(image,sigma_s=8,sigma_r=0.1,shade_factor=0.010)
plt.subplot(121)
plt.imshow(pencil_image_0)
plt.title('Imagen Dibujo_0')
plt.axis('off')
plt.subplot(122)
plt.imshow(pencil_image_1)
plt.title('Imagen Dibujo_1')
plt.axis('off')
plt.show()


# In[3]:


image_style=cv2.stylization(image,100,0.5)
plt.imshow(image_style)
plt.title('Image style')
plt.axis('off')
plt.show()


# In[4]:


# Uniendo los 2 filtros

image_cartooning_1=cv2.stylization(pencil_image_0,170,0.50)
image_cartooning_0=cv2.stylization(pencil_image_1,10,0.99)

plt.subplot(121)
plt.imshow(image_cartooning_1)
plt.title('Cartooning Image_1')
plt.axis('off')
plt.subplot(122)
plt.imshow(image_cartooning_0)
plt.title('Cartooning Image_0')
plt.axis('off')
plt.show()


# In[5]:


# Variante de cartooning image

image_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
blur_image=cv2.medianBlur(image_gray,5)
laplacian_image=cv2.Laplacian(blur_image,cv2.CV_8U,ksize=5,scale=1,delta=0.5)
ret,threshold_image=cv2.threshold(laplacian_image,10,50,cv2.THRESH_BINARY_INV)
bilateral_image=cv2.bilateralFilter(image_gray,10,250,250)
cartooning_image=cv2.bitwise_and(bilateral_image,bilateral_image,mask=threshold_image)
plt.subplot(121)
plt.imshow(image)
plt.title('Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(cartooning_image)
plt.title('Cartooning')
plt.axis('off')
plt.show()


# In[ ]:




