#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,15)


# In[4]:


# Transformaciones Morfologicas
# Dilatacion / Erosion

image_binaria=cv2.imread('Maikel_Gray.jpg')
dilation=cv2.dilate(image_binaria,kernel=(5,5),iterations=3)
erosion=cv2.erode(image_binaria,kernel=(5,5),iterations=3)
plt.subplot(131)
plt.imshow(image_binaria)
plt.title('Image binaria')
plt.axis('off')
plt.subplot(132)
plt.imshow(dilation)
plt.title('Image dilation')
plt.axis('off')
plt.subplot(133)
plt.imshow(erosion)
plt.title('Image erosion')
plt.axis('off')
plt.show()


# In[5]:


# Operaciones de apertura
# Esta operacion hace uso de una erosion seguido de una dilatacion usando la misma extructura del kernel
opening=cv2.morphologyEx(image_binaria,cv2.MORPH_OPEN,kernel=(5,5))
plt.subplot(121)
plt.imshow(image_binaria)
plt.title('Image binaria')
plt.axis('off')
plt.subplot(122)
plt.imshow(opening)
plt.title('Image opening')
plt.axis('off')
plt.show()


# In[6]:


# Operacion cerrado
# Esta operacion elige una dilatacion seguido de una erosion con el mismo kernel
# la operacion de dilatacion llena pequños basios en imagenes 

closing=cv2.morphologyEx(image_binaria,cv2.MORPH_CLOSE,kernel=(5,5))

plt.subplot(121)
plt.imshow(image_binaria)
plt.title('Image binaria')
plt.axis('off')
plt.subplot(122)
plt.imshow(closing)
plt.title('Image closing')
plt.axis('off')
plt.show()


# In[7]:


# Operacion morfologica del gradiente
# Se define como la diferencia entre dilatacion y erosion en una imagen de entrada

morph_gradient=cv2.morphologyEx(image_binaria,cv2.MORPH_GRADIENT,kernel=(5,5))

plt.subplot(121)
plt.imshow(image_binaria)
plt.title('Image binaria')
plt.axis('off')
plt.subplot(122)
plt.imshow(morph_gradient)
plt.title('Image gradient')
plt.axis('off')
plt.show()


# In[8]:


# Operacion top de sombrero
# Se define como la deferencica entre la imagen de entrada y la imagen de apertura 

top_hat=cv2.morphologyEx(image_binaria,cv2.MORPH_TOPHAT,kernel=(5,5))
plt.subplot(121)
plt.imshow(image_binaria)
plt.title('Image binaria')
plt.axis('off')
plt.subplot(122)
plt.imshow(top_hat)
plt.title('Image top_hat')
plt.axis('off')
plt.show()


# In[9]:


# Operacion de sombrero negro
# Se define como la diferencia entre la imagen de entrada y operacion closing

black_hat=cv2.morphologyEx(image_binaria,cv2.MORPH_BLACKHAT,kernel=(5,5))
plt.subplot(121)
plt.imshow(image_binaria)
plt.title('Image binaria')
plt.axis('off')
plt.subplot(122)
plt.imshow(black_hat)
plt.title('Image black_hat')
plt.axis('off')
plt.show()


# In[10]:


# Elementos estructurados
# La funcion  getStructuringElement puede usar en las operaciones morphologicas para hacer kernel personalizados
# Tres kernel ofrece OpenCV: cv2.MORPH_RECT,cv2.MORPH_ELLIPSE,cv2.MORPH_CROSS

kernel=cv2.getStructuringElement(shape=cv2.MORPH_CROSS,ksize=(5,5))
image_valor=cv2.morphologyEx(image_binaria,op=cv2.MORPH_CROSS,kernel=kernel)
plt.subplot(121)
plt.imshow(image_binaria)
plt.title('Image binaria')
plt.axis('off')
plt.subplot(122)
plt.imshow(image_valor)
plt.title('Image kernel personalizado')
plt.axis('off')
plt.show()


# ***Nota: Las operaciones morphologicas son usadas cuando se procesan imagenes porque podemos elimiar algun tipo de ruido en la imagen,lo cual puede ser propio del proceso de procesado. Tambien pueden ser usadas para solucionar imperfecciones en la extructura de la imagen ***
