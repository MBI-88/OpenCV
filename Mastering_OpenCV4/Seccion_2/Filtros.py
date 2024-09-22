#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,15)


# In[3]:


# Filtros de imagenes
image=cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/cat-02.jpg')
kernel_averaging_5x5=np.ones((5,5),np.float32)/25 # El valor que se tendra es 0.04

smooth_image_f2D=cv2.filter2D(image,-1,kernel_averaging_5x5)
#smooth_image_f2D=cv2.blur(image,ksize=(5,5)) Hace la misma funcion  
plt.imshow(smooth_image_f2D)
plt.title('Imagen filtrada')
print('Forma: ',smooth_image_f2D.shape)
plt.show()


# In[4]:


# Promediando filtros
image=cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/cat-03.jpg')
image=image[:,:,::-1]
smooth_image_blur=cv2.blur(image,(10,10))
smooth_image_bfi=cv2.boxFilter(image,-1,(10,10),normalize=False)# Si normalize es True entonces es igual a blur
plt.subplot(121)
plt.imshow(smooth_image_blur)
plt.title('Filtro blur')
plt.subplot(122)
plt.imshow(smooth_image_bfi)
plt.title('Filtro bif')
print('Forma blur: {}  Forma bfi: {}'.format(smooth_image_blur.shape,smooth_image_bfi.shape))
plt.show()


# In[5]:


# Filtros Gaussianos
smooth_image_gb=cv2.GaussianBlur(image,(9,9),0.5,0.5)
plt.imshow(smooth_image_gb)
plt.title('Filtro Gaussiano')
print('Forma: ',smooth_image_gb.shape)
plt.show()


# In[6]:


# Mediana Filtros
smooth_image_mb=cv2.medianBlur(image,5)
plt.imshow(smooth_image_mb)
plt.title('Filtro mediana')
plt.show()
# Es usado para reducir el ruido de sal y pimienta en un imagen


# In[7]:


# Filtro bilateral
# Se usa para reducir el ruido mientras se mantiene la forma de los bordes

smooth_image_bfl=cv2.bilateralFilter(image,d=5,sigmaColor=10,sigmaSpace=10)
plt.subplot(121)
plt.imshow(image)
plt.title('Image')
plt.subplot(122)
plt.imshow(smooth_image_bfl)
plt.title('Filtro bilateral')
plt.show()


# In[8]:


# Afinando imagenes

unsharped=cv2.addWeighted(image,1.1,smooth_image_gb,-0.45,10)
plt.imshow(unsharped)
plt.title('Unsharped image')
plt.show()


# In[9]:


# Filtros usados en imagenes

f2D=cv2.filter2D(image,-1,(2,2),2)
sobel=cv2.Sobel(image,-1,1,1,(2,2))
scharr=cv2.Scharr(image,-1,0,1,(2,2),2)
laplaciano=cv2.Laplacian(image,-1,2,1,10)

plt.subplot(151)
plt.imshow(f2D)
plt.title('Filtro2D ')
plt.subplot(152)
plt.imshow(sobel)
plt.title('Sobel')
plt.subplot(153)
plt.imshow(scharr)
plt.title('Scharr')
plt.subplot(154)
plt.imshow(laplaciano)
plt.title('Laplaciano')
plt.subplot(155)
plt.imshow(image)
plt.title('Image')
plt.show()


# In[ ]:


# Uso de la funcion getGaussian
kernel_x=cv2.getGaussianKernel(3,1)/9
kernel_y=cv2.getGaussianKernel(3,1)
foto=cv2.sepFilter2D(image,-1,kernel_x,kernel_y,2)
plt.imshow(foto)
plt.title('Filtro gausssiano personalizado')
plt.show()


# In[ ]:




