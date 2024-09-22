#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,15)


# In[2]:


# Escalado de imagenes
image=cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/dog-02.jpg')
image=image[:,:,::-1]
w,h=image.shape[:2]
resized_image=cv2.resize(image,(w*2,h*2),interpolation=cv2.INTER_LINEAR)
plt.subplot(121)
plt.imshow(resized_image)
plt.title('Image resized')
plt.subplot(122)
plt.imshow(image)
plt.title('Image original')
print('Image risized form: ',resized_image.shape)
print('Image original form: ',image.shape)
plt.show()


# In[3]:


# Variante al metodo anterior
dst_image=cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
plt.imshow(dst_image)
plt.title('Imagen cortada por 2')
print('Forma: ',dst_image.shape)
plt.show()


# ***Nota: Para agrandar la imagen la mejor forma es usando cv2.INTER_CUBIC o cv2.INTER_LINEAR. Si se desea cortar la imagen el mejor metodo de interpolacion es cv2.INTER_LINEAR***

# In[4]:


# Traslacion de imagen
# M=np.float32([[1,0,x],[0,1,y]])
M=np.float32([[1,0,200],[0,1,30]])
image_translating=cv2.warpAffine(image,M,image.shape[:2])
plt.imshow(image_translating)
plt.title('Imagen trasladada')
plt.show()


# In[5]:


# Traslacion negativa
M=np.float32([[1,0,-200],[0,1,-100]])
image_translating_neg=cv2.warpAffine(image,M,image.shape[:2])
plt.imshow(image_translating_neg)
plt.title('Imagen trasladada (negativa)')
plt.show()


# In[6]:


# Rotacion de imagen
M=cv2.getRotationMatrix2D((image.shape[0]/2.0,image.shape[1]/2.0),angle=90,scale=1)
dst_image_rotate=cv2.warpAffine(image,M,image.shape[:2])
plt.imshow(dst_image_rotate)
plt.title('Imagen rotada')
plt.show()


# In[7]:


# Afinando la transformacion de una imagen
pts_1=np.float32([[225,0],[350,9],[90,210]])
pts_2=np.float32([[100,30],[190,45],[15,300]])
M=cv2.getAffineTransform(pts_1,pts_2)
dst_image=cv2.warpAffine(image,M,image.shape[:2])
plt.imshow(dst_image)
plt.title('Imagen afinada')
print('Forma: ',dst_image.shape)
plt.show()


# In[8]:


# Transformacion de perspectiva  de imagen
pts_1=np.float32([[420,69],[509,69],[422,140],[551,140]])
pts_2=np.float32([[0,0],[300,0],[0,300],[300,315]])
M=cv2.getPerspectiveTransform(pts_1,pts_2)
dst_image=cv2.warpPerspective(image,M,(800,800))
plt.imshow(dst_image)
plt.title('Imagen perspectiva')
print('Forma: ',dst_image.shape)
plt.show()


# In[9]:


#  Cropping image

image_cropp=image[200:800,60:1150]
plt.imshow(image_cropp)
plt.title('Imagen cropp')
print('Forma: ',image_cropp.shape)
plt.show()


# In[ ]:




