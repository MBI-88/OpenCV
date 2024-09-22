#!/usr/bin/env python
# coding: utf-8

# ***Tecnicás para procesado de imagenes***

# In[2]:


# Separando y mesclando imagenes por calanes
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,15)


# In[2]:


image=cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/dog-02.jpg')
(b,g,r)=cv2.split(image)
# Variante mas optima, necesita menos tiempo de computo
B=image[:,:,0]
G=image[:,:,1]
R=image[:,:,2]
plt.subplot(131)
plt.title('Blue')
plt.axis('off')
plt.imshow(B)
plt.subplot(132)
plt.title('Green')
plt.axis('off')
plt.imshow(G)
plt.subplot(133)
plt.title('Red')
plt.axis('off')
plt.imshow(r)
plt.show()


# In[3]:


# Mesclando los canales en una misma imagen
image_copy=cv2.merge((r,g,b)) # Usando el formato RGB
#cv2.imshow('RGB',image_copy)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
image_copy_1=cv2.merge((b,g,r))
plt.subplot(121)
plt.imshow(image_copy)
plt.title('RGB')
plt.axis('off')
plt.subplot(122)
plt.imshow(image_copy_1)
plt.title('BGR')
plt.axis('off')
plt.show()


# In[4]:


# Apagando colores en imagenes

image_without_green=image.copy()
image_without_green[:,:,1]=0

plt.imshow(image_without_green)
plt.title('Imagen sin verde',fontdict={'fontsize':15})
plt.axis('off')
plt.show()

