#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,15)


# In[2]:


# Aritmetica de imagenes

x=np.uint8([250])
y=np.uint8([50])
result_cv=cv2.add(x,y)
# 250 + 50 = 300  => 255
print("cv2.add(x:'{}', y:'{}') = '{}'".format(x,y,result_cv))
result_numpy=x+y
# 250 + 50  = 300 % 256 = 44
print("x:'{}' + y:'{}' = '{}'".format(x,y,result_numpy))
# Nota: En opencv  los valores son cortados en el rango de 0--255 (saturacion)
# en numpy son arrastrados por medio del redondeo (operacion modulo)


# In[4]:


image=cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/cat-01.jpg')

M=np.ones_like(image[:,:,::-1]) * 60 # Para adicionar 60 a todos los pixeles de una imagen
added_image=cv2.add(image[:,:,::-1],M)
plt.imshow(added_image)
plt.title('Imagen sumada 60')
plt.axis('off')
plt.show()


# In[5]:


# Variante  
scalar=np.ones((1,3),dtype='float') * 110
added_image_2=cv2.add(image[:,:,::-1],scalar)
plt.imshow(added_image_2)
plt.title('Imagen sumada 110')
plt.axis('off')
plt.show()


# In[6]:


subtract_image=cv2.subtract(image[:,:,::-1],M)
plt.imshow(subtract_image)
plt.title('Imagen sustraida 60')
plt.axis('off')
plt.show()


# In[ ]:




