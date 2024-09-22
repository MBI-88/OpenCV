#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:06:51 2021

@author: mbi
"""
import cv2 
import numpy as np
import matplotlib.pyplot as plt
#%%
# Entendimineto de cv2.dnn.blobFromImage()
"""
Parametros de la funcion:
    image: imagen de entrada para procesar
    scalefactor: multiplicador para los valores de la imagen
    size: tamño de la imagen de salida
    mean: escalar con valores mendios sustraidos de la imagen.Si se elige sustraccion media los valores son entendidos como (mean-R,mean-G,mean-B) cuando se utiliza swapRB=True
    swapRB: esta bandera puede ser usada para cambiar los canales R y B cuando se pone en True
    crop: esta bandera indica si la imagen sera agrandada despues se un ajuste de tamaño.
    ddepth: es la profundidad del blob. Se puede elegir entre CV_32F o CV_8U.
    
"""
#%%

def get_image_from_blob(blob_image,scalefactor,dim,mean,swap_rb,mean_added):
    images_from_blob = cv2.dnn.imagesFromBlob(blob_image)
    image_from_blob = np.reshape(images_from_blob[0],dim) / scalefactor
    image_from_blob_mean = np.uint8(image_from_blob)
    image_from_blob = image_from_blob_mean + np.uint8(mean)
    
    if mean_added is True:
        if swap_rb:
            image_from_blob = image_from_blob[:,:,::-1]
        return image_from_blob
    else:
        if swap_rb:
            image_from_blob_mean = image_from_blob_mean[:,:,::-1]
        return image_from_blob_mean

def show_with_matplolib(image,title,pos):
    image_rgb = image[:,:,::-1]
    plt.subplot(2,2,pos)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    
    
image = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/face_test.jpg')

blob_image = cv2.dnn.blobFromImage(image,1.0,(300,300),[104.,117.,123.],False,False)

img_from_blob = get_image_from_blob(blob_image,1.0,(300,300,3),[104.,117.,123.],False,True)
img_from_blob_swap = get_image_from_blob(blob_image,1.0,(300,300,3),[104.,117.,123.],True,True)
img_from_blob_mean = get_image_from_blob(blob_image,1.0,(300,300,3),[104.,117.,123.],False,False)
img_from_blob_mean_swap = get_image_from_blob(blob_image,1.0,(300,300,3),[104.,117.,123.],True,False)

fig = plt.figure(figsize=(12,5))
plt.suptitle("cv2.dnn.blobFromImage() visualizacion",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

show_with_matplolib(img_from_blob, "Img from blob",1)
show_with_matplolib(img_from_blob_swap,"Img from blob swap",2)
show_with_matplolib(img_from_blob_mean,"Img from blob mean",3)
show_with_matplolib(img_from_blob_mean_swap,"Img from blob mean swap",4)

plt.show()
#%%
def get_images_from_blob(blob_images,scalefactor,dim,mean,swap_rb,mean_added):
    images_from_blob = cv2.dnn.imagesFromBlob(blob_images)
    imgs = []
    
    for image_blob in images_from_blob:
        image_from_blob = np.reshape(image_blob,dim) / scalefactor
        image_from_blob_mean = np.uint8(image_from_blob)
        image_from_blob = image_from_blob_mean + np.uint8(mean)
        if mean_added is True:
            if swap_rb:
                image_from_blob = image_from_blob[:,:,::-1]
            imgs.append(image_from_blob)
        else:
            if swap_rb:
                image_from_blob_mean = image_from_blob_mean[:,:,::-1]
            imgs.append(image_from_blob_mean)
    return imgs

def show_with_matplolib(image,title,pos):
    image_rgb = image[:,:,::-1]
    plt.subplot(2,4,pos)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')

image2 = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/face_test_2.jpg')
images = [image,image2]

blob_images = cv2.dnn.blobFromImages(images,1.0,(300,300),[104.,117.,123.],False,False)
print(blob_images.shape,"\n")

imgs_from_blob = get_images_from_blob(blob_images,1.0,(300,300,3),[104.,117.,123.],False,True)
imgs_from_blob_mean = get_images_from_blob(blob_images,1.0,(300,300,3),[104.,117.,123.],False,False)
imgs_from_blob_swap = get_images_from_blob(blob_images,1.0,(300,300,3),[104.,117.,123.],True,True)
imgs_from_blob_mean_swap = get_images_from_blob(blob_images,1.0,(300,300,3),[104.,117.,123.],True,False)

fig = plt.figure(figsize=(12,5))
fig.patch.set_facecolor("blue")
plt.suptitle("cv2.dnn.blobFromImages() visualizacion",fontsize=14,fontweight='bold')

show_with_matplolib(imgs_from_blob[0],"Img from blob",1)
show_with_matplolib(imgs_from_blob_swap[0],"Img from blob swap",2)
show_with_matplolib(imgs_from_blob_mean[0],"Img from blob_mean",3)
show_with_matplolib(imgs_from_blob_mean_swap[0],"Img from blob mean swap",4)

show_with_matplolib(imgs_from_blob[1],"Img from blob",5)
show_with_matplolib(imgs_from_blob_swap[1],"Img from blob",6)
show_with_matplolib(imgs_from_blob_mean[1],"Img from blob",7)
show_with_matplolib(imgs_from_blob_mean_swap[1],"Img from blob",8)

plt.show()

    
#%%
def get_cropped_image(image):
    size = min(image.shape[0],image.shape[1])
    x1 = int(0.5 * (image.shape[1] - size))
    y1 = int(0.5 * (image.shape[0] - size))
    return image[y1:(y1 + size),x1:(x1 + size)]

cropped_image = get_cropped_image(image)
print("Shape cropped: {}\n".format(cropped_image.shape))

blob_images =  cv2.dnn.blobFromImages(images,1.0,(300,300),[104.,117.,123.],False,False)
blob_images_cropped = cv2.dnn.blobFromImages(images,1.0,(300,300),[104.,117.,123.],False,True)

imgs_from_blob = get_images_from_blob(blob_images,1.0,(300,300,3),[104.,117.,123.],False,False)
imgs_from_blob_crop = get_images_from_blob(blob_images_cropped,1.0,(300,300,3),[104.,117.,123.],False,True)

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor("green")
plt.suptitle("cv2.dnn.blobFromImages() visualizacion",fontsize=14,fontweight='bold')

show_with_matplolib(imgs_from_blob[0],"Image origianl",1)
show_with_matplolib(imgs_from_blob[1],"Image origianl",2)
show_with_matplolib(imgs_from_blob_crop[0],"Image cropped",3)
show_with_matplolib(imgs_from_blob_crop[1],"Image cropped",4)

plt.show()


#%%
    

