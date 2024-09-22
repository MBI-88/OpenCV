#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


imagen=cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/cat-03.jpg')
b,g,r=cv2.split(imagen)
imagen_matplolib=cv2.merge([r,g,b])


# In[4]:


plt.subplot(121)
plt.imshow(imagen)
plt.axis('off')
plt.subplot(122)
plt.imshow(imagen_matplolib)
plt.axis('off')
plt.show()


# In[5]:


# Concatenando imagenes en una misma ventana
img_concat = np.concatenate((imagen,imagen_matplolib),axis=1)
cv2.imshow('BGR image and RGB image',img_concat)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


# Separando imagen por canales con numpy similar a cv2.split

B=imagen_matplolib[:,:,0]
G=imagen_matplolib[:,:,1]
R=imagen_matplolib[:,:,2]
# Ejemplo de salida
plt.imshow(B)
plt.axis('off')
plt.show()


# In[8]:


# Convirtiendo de BGR a RGB con matplotlib

new_imagen_matplot=imagen[:,:,::-1]
plt.imshow(new_imagen_matplot)
plt.axis('off')
plt.show()


# ***Capturando archivos de imagenes***

# In[9]:


# Usando del modulo sys.argv para captura de lineas de comandos
import sys,argparse
print('El nombre del scripts que esta procesandose es : '.format(sys.argv[0]))
print('El numero de argumento del scripts es : '.format(sys.argv[1]))
print('El argumento del scripts es : '.format(sys.argv[2])) 
# Esto  es para usarlo por consola el scripts


# In[10]:


# Usando argaparse para pasar lineas de comando mas complejos

parser = argparse.ArgumentParser()
parser.add_argument('primer_argumento',help='este es un texto string en coneccion con el primer argumento')
arg=parser.parse_args()
print(arg.first_argument)
# Es para usarse por consola

