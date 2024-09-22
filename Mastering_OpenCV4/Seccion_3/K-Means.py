# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:48:35 2021

@author: MBI
"""
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter
#%%
# K-means clustering

"""
Opencv provee la funcion cv2.kmeans() la cual implementa el algoritmo de vecinos mas cercanos
el cual  encuentra centros de clusters y grupos entre los datos de entrada.

Definicion de la funcion : retval,bestlavel,centers = cv2.kmeans(data,K,bestLabels,criteria,attempts,
                                                                 flags,centers)
data: representa el dato de entrada para clusterizar debe ser np.float32 y se debe redimensionar
K: especifica el numerode clusters requeridos para la salida.
criteria: parametro que establece el maximo numeros de iteraciones o presicion deseada. Esta compuesta de una tupla:
    type: tipo  de criterio de terminacion, tiene 3 banderas:
        -cv2.TERM_CRITERIA_EPS: El algoritmo para cuando la presicion es alcanzada.
        -cv2.TERM_CRTTERIA_MAX_ITER: El algoritmo para cuando se alcanza el final de las iteraciones
        -cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER: El algoritmo para cuando se cumple una de las dos anteriores
    
    max_iter: Maximo numero de iteraciones
    epsilon: Presicion requerida.

attempts: especifica  el numero de veces que el algoritmo es ejecutado usando diferentes etiquetas iniciales.
flags: especifica cuantos centros iniciales son tomados. El cv2.KMEANS_RANDOM_CENTERS selecciona centros aleatorios iniciales
      El cv2.KMEANS_PP_CENTERS usa la inicializacion  de centros k-mean++

La funcion retorna:
    bestLabels : un arreglo de enteros que almacena el indice del cluster para cada muestra.
    centers: un arreglo que contiene el centro para cada cluster.
    compactness: es la suma del cuadrado de la distancia para cada uno de los puntos con sus respectivos centros.
    
"""
#%%
data = np.float32(np.vstack(
    (np.random.randint(0,40,size=(50,2)),np.random.randint(30,70,(50,2)),np.random.randint(60,100,(50,2)))))

# Cluster-(K=2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,10)

ret,label,center = cv2.kmeans(data,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

A = data[label.ravel() == 0]
B = data[label.ravel() == 1]

fig = plt.figure(figsize=(12,6))
fig.patch.set_facecolor('silver')
plt.suptitle('K-Means clustering algorithm')

ax = plt.subplot(1,2,1)
plt.scatter(data[:,0],data[:,1],c='c',marker='v')
plt.title('Data')

ax = plt.subplot(122)
plt.scatter(A[:,0],A[:,1],c='b',marker='p')
plt.scatter(B[:,0],B[:,1],c='g',marker='*')
plt.scatter(center[:,0],center[:,1],s=100,c='m',marker='s')
plt.title('Clustered data and centroids (K= 2)')
plt.show()
#%%
# Cluster-(K=3)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,10)

ret,label,center = cv2.kmeans(data,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

A = data[label.ravel() == 0]
B = data[label.ravel() == 1]
C = data[label.ravel() == 2]

fig = plt.figure(figsize=(12,6))
fig.patch.set_facecolor('silver')
plt.suptitle('K-Means clustering algorithm')

ax = plt.subplot(1,2,1)
plt.scatter(data[:,0],data[:,1],c='c',marker='v')
plt.title('Data')

ax = plt.subplot(122)
plt.scatter(A[:,0],A[:,1],c='b',marker='p')
plt.scatter(B[:,0],B[:,1],c='g',marker='*')
plt.scatter(C[:,0],C[:,1],c='y',marker='s')
plt.scatter(center[:,0],center[:,1],s=100,c='m',marker='s')
plt.title('Clustered data and centroids (K = 3)')
plt.show()


#%%
# Cluster-(K=4)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,10)

ret,label,center = cv2.kmeans(data,4,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

A = data[label.ravel() == 0]
B = data[label.ravel() == 1]
C = data[label.ravel() == 2]
D = data[label.ravel() == 3]

fig = plt.figure(figsize=(12,6))
fig.patch.set_facecolor('silver')
plt.suptitle('K-Means clustering algorithm')

ax = plt.subplot(1,2,1)
plt.scatter(data[:,0],data[:,1],c='c',marker='v')
plt.title('Data')

ax = plt.subplot(122)
plt.scatter(A[:,0],A[:,1],c='b',marker='p')
plt.scatter(B[:,0],B[:,1],c='g',marker='*')
plt.scatter(C[:,0],C[:,1],c='y',marker='s')
plt.scatter(D[:,0],D[:,1],c='r',marker='o')
plt.scatter(center[:,0],center[:,1],s=100,c='m',marker='s')
plt.title('Clustered data and centroids (K = 4)')
plt.show()
#%%
# Cuantizacion de color usando K-meanz clustering

landcascade = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/landscape_1.jpg')

def color_quantization(img,k):
    data = np.float32(img).reshape(-1,3)# Es necesario reformar la imagen para que no se pixele
    print(data.shape)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,20,1.0)
    
    ret,label,center = cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

color_q3 = color_quantization(landcascade,3)
color_q5 = color_quantization(landcascade,5)
color_q10 = color_quantization(landcascade,10)
color_q20 = color_quantization(landcascade,20)
color_q40 = color_quantization(landcascade,40)

fig = plt.figure(figsize=(12,5))
fig.patch.set_facecolor('silver')
plt.suptitle('Color quantization using K-means',fontsize=12,fontweight='bold')
plt.subplot(231)
plt.imshow(landcascade[:,:,::-1])
plt.title('Original Image')
plt.axis('off')
plt.subplot(232)
plt.imshow(color_q3[:,:,::-1])
plt.title('color quantization k = 3')
plt.axis('off')
plt.subplot(233)
plt.imshow(color_q5[:,:,::-1])
plt.title('color quantization k = 5')
plt.axis('off')
plt.subplot(234)
plt.imshow(color_q10[:,:,::-1])
plt.title('color quantization k = 10')
plt.axis('off')
plt.subplot(235)
plt.imshow(color_q20[:,:,::-1])
plt.title('color quantization k = 20')
plt.axis('off')
plt.subplot(236)
plt.imshow(color_q40[:,:,::-1])
plt.title('color quantization k = 40')
plt.axis('off')
plt.show()
    
#%%

def color_quantization_dis(img,k):
    data = np.float32(img).reshape(-1,3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,20,1.0)
    ret,label,center = cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    print("Center shape {} Label shape {}".format(center.shape,label.shape))
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    
    counter = Counter(label.flatten())
    print(counter)
    
    total = img.shape[0] * img.shape[1]
    desired_height = 70
    desired_width = img.shape[1]
    desired_height_color = 50
    
    color_distribution = np.ones((desired_height,desired_width,3),dtype='uint8') * 255
    
    start = 0
    for key,value in counter.items():
        value_normalized = value / total * desired_width
        end = start + value_normalized
        cv2.rectangle(color_distribution,(int(start),0),(int(end),desired_height_color),center[key].tolist(),-1)
        start = end
    return np.vstack((color_distribution,result))
    
landcascade_2 = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/landscape_2.jpg')

color_q3 = color_quantization_dis(landcascade_2,3)
color_q5 = color_quantization_dis(landcascade_2,5)
color_q10 = color_quantization_dis(landcascade_2,10)
color_q20 = color_quantization_dis(landcascade_2,20)  
color_q40 = color_quantization_dis(landcascade_2,40)


fig = plt.figure(figsize=(12,5))
fig.patch.set_facecolor('silver')
plt.suptitle('Color quantization using K-means',fontsize=12,fontweight='bold')
plt.subplot(231)
plt.imshow(landcascade_2[:,:,::-1])
plt.title('Original Image')
plt.axis('off')
plt.subplot(232)
plt.imshow(color_q3[:,:,::-1])
plt.title('color quantization k = 3')
plt.axis('off')
plt.subplot(233)
plt.imshow(color_q5[:,:,::-1])
plt.title('color quantization k = 5')
plt.axis('off')
plt.subplot(234)
plt.imshow(color_q10[:,:,::-1])
plt.title('color quantization k = 10')
plt.axis('off')
plt.subplot(235)
plt.imshow(color_q20[:,:,::-1])
plt.title('color quantization k = 20')
plt.axis('off')
plt.subplot(236)
plt.imshow(color_q40[:,:,::-1])
plt.title('color quantization k = 40')
plt.axis('off')
plt.show()
#%%




