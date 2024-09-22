#!/usr/bin/env python
# coding: utf-8

# ***Intriduccion a openCv, manipulacion de las propiedades de una imagen***
import cv2
import numpy as np
import matplotlib.pyplot as plt

# In[1]:
    
image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/cat-01.jpg')

dimension = image.shape
total_element = image.size
image_dtupe = image.dtype

print("Deimesion: {} Tamaño: {} Tipo: {}".format(dimension,total_element,image_dtupe))


# In[2]:


cv2.imshow('Gato',image)
cv2.waitKey(10000) # Funcion que espera por la interrupcion de teclado cuando es 0 el argumento en caso contrario el tiempo es en milisegundos

cv2.destroyAllWindows()
# In[3]:


(b,g,r)=image[6,40] # Accediendo a pixeles especificos de una imagen 
print(b,' ',g,' ',r)


# In[4]:


B=image[6,40,0] # Accediendo a un canal a la vez
print(B)


# In[5]:


image[6,40]=(0,0,255) # Modificando el valor de un pixel (r,g,b)
cv2.imshow('Gato_modificado',image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
# In[6]:


region_image=image[0:100,0:100] # Accediendo a una region de la imagen
cv2.imshow('Region_Imagen',region_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()

# In[2]:


image_gray=cv2.imread('C:/Users/MBI/Pictures/Camera Roll/Maikel_Gray.jpg',cv2.IMREAD_GRAYSCALE) # Cargando una imagen en escala de grices


# In[3]:


imagen_gray_cp=image_gray.copy()
dimension = image_gray.shape
tamaño = image_gray.size
dato = image_gray.dtype

print('Estructura ',dimension)
print('Tamaño ',tamaño)
print('Tipo ',dato)


# In[4]:


intesidad_pixel = imagen_gray_cp[6,40]
intesidad_pixel


# In[5]:


imagen_gray_cp[6,40]=0 # Cambiando la intensidad de un pixel a negro (0) maximo brillo (1)
cv2.imshow('Maikel_modificado',imagen_gray_cp)


# In[7]:


imagen=cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/cat-03.jpg')
b,g,r=cv2.split(imagen)


# In[8]:


imagen_matplolib=cv2.merge([r,g,b])


# In[ ]:




