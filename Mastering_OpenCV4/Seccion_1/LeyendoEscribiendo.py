#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2 
import matplotlib.pyplot as plt
import numpy as np
import argparse


# In[4]:


parser=argparse.ArgumentParser()
parser.add_argument('Direccion_imagen',help='direccion para importar la imagen')
args=parser.parse_args() # Opteniendo la imagen a travez de args
image=cv2.imread(args.Drireccion_imagen)
args=vars(parser.parse_args()) # Opteniendo la imagen a travez de un diccionario
image2=cv2.imread(args['Direccion_imagen'])

cv2.imshow('Load_Image',image)
cv2.imshow('Load_Image2',image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Es para  cargar la imagen por consola


# In[5]:


# Cambiando la imagen a escala de grices
image= cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/cat-03.jpg')
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagen_gray',image_gray )
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('C:/Users/MBI/Documents/Python_Scripts/Master_OpenCV4/Gato_gray.jpg',image_gray)


# In[6]:


# Leyendo frames de camaras
captura=cv2.VideoCapture(0)

frame_width=captura.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height=captura.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps=captura.get(cv2.CAP_PROP_FPS)
print('Frame width ',frame_width)
print('Frame height ',frame_height)
print('FPS ',fps)

if captura.isOpened() is False:
    print('Error opening camera')

while captura.isOpened():
    ret,frame=captura.read()
    if ret is True:
        cv2.imshow('Input Camera',frame)
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray input camera',gray_frame)
        if cv2.waitKey(20) & 0xFF == ord('q') : # & es igual a and
            break
    else:
        break
captura.release()
cv2.destroyAllWindows()


# In[ ]:


# Guardando frame de camaras

if cv2.waitKey(20) & 0xFF == ord('c'):
    frame_name='Camera_frame_{}.png'.format(frame_index)
    gray_frame_name='Gray_frame_{}.png'.format(frame_index)
    cv2.imwrite(frame_name,frame)
    cv2.imwrite(gray_frame_name,gray_frame)
    frame_index += 1


# In[ ]:


# Leyendo  archivo de videos desde camara Ip
ip_captura=cv2.VideoCapture('http://217.126.89.102:8010/axis-cgi/mjpg/video.cgi')
ip_captura.read()


# In[ ]:


# Escribiendo un archivo de video
video_cap=cv2.VideoCapture(0)
frame_width=video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height=video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps=video_cap.get(cv2.CAP_PROP_FPS)
print('Frame width ',frame_width)
print('Frame height ',frame_height)
print('FPS ',fps)
code=cv2.VideoWriter_fourcc(*'DIVX') # Formatos de code: DIVX,XVID,X256 y MJPG
                                     # Formatos de compresion: avi,mp4,mov,wmv y mpj

output_gray=cv2.VideoWriter('Video_gray.avi',code,int(fps),(int(frame_width),int(frame_height)),False) # El ultimo argumento es falso porque es en escala de grices
while video_cap.isOpened():
    ret,frame=video_cap.read() # Se usa para leer framme
    if ret is True :
        gray_video=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        output_gray.write(gray_video)
        cv2.imshow('Gray_Video',gray_video)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video_cap.release()
output_gray.release()
cv2.destroyAllWindows()


# In[7]:


# Opteniendo todas las propiedades de archivos de video
def decode_fourcc(fourcc):
    "Decodifica el valor por parametro opteniendo el char"
    fourcc_int=int(fourcc)
    fourcc_decode=''
    for i in range(4):
        int_value=fourcc_int >> 8 * i & 0xFF
        print('int_value: {}'.format(int_value))
        fourcc_decode += chr(int_value)
    return fourcc_decode

captura=cv2.VideoCapture('Video_gray.avi')
fourcc=captura.get(cv2.CAP_PROP_FOURCC)
decode_fourcc(fourcc)


# In[ ]:




