# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:38:39 2021

@author: MBI
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
#%%
# K-NN
"""
El algoritmo es usado para problemas de regresion como de clasificacion, este metodo trabaja de la 
siguiente forma :
    Fase de entrennamiento: KNN almacena vectores de variables asi como etiquetas de clases de todas las
    muestras de entrenamiento.
    
    Fase de estimacion: Un vector no etiquetado es clasificado como la etiqueta de clase de mayor ocurrencia  
    entre las k muestras de entrenamiento mas cercanas al vector que sera clasificado,donde k es una constante 
    definida por el usuario.

Opencv provee la funcion cv2.ml.KNearest_create() para el uso del algoritmo KNN
Esta funcion crea un KNN basio el cual sera entrenado usando el metodo train() al cual se 
le dan los datos y las etiquetas. Finalmente esl metodo findNearest() es usado para encontrar los vecinos.

Definicion de la funcion:
    retval,result,neighborResposes,dist = cv2.ml.KNearest.findNearest(sample,k,[,result[,neighborResposes[,dist]]])
    donde:
        smaples: son las muestras almacenadas por filas.
        k: establece el numero de vecinos mas cercanos (k > 1).
        result: almacena las predicciones para cada una de las muestras de entrada.
        neighborResposes: almacena los correspondientes vecinos.
        dist: almacena las distancias de las muestras de entrada hcacia los correspondientes vecinos.
        
"""
#%%
# Entendimiento de KNN

data = np.random.randint(0,100,(16,2)).astype(np.float32)
labels = np.random.randint(0,2,(16,1)).astype(np.float32)

sample = np.random.randint(0,100,(1,2)).astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(data,cv2.ml.ROW_SAMPLE,labels)
k = 3
ret,result,neighbours,dist = knn.findNearest(sample,k)


print("result: ",result)
print("neighbours: ",neighbours)
print("dist: ",dist)

fig = plt.figure(figsize=(8,6))
fig.patch.set_facecolor('silver')

red_triangle = data[labels.ravel() == 0]
blue_squares = data[labels.ravel() == 1]
plt.scatter(blue_squares[:,0],blue_squares[:,1],s=200,c='b',marker='s') 
plt.scatter(red_triangle[:,0],red_triangle[:,1],s=200,c='r',marker='^')
plt.scatter(sample[:,0],sample[:,1],200,'g','o')

if result[0][0] > 0:
    plt.suptitle("k-NN algorithm: sample green point is classified as blue (k = " + str(k) + ")", fontsize=14,
                 fontweight='bold')
else:
    plt.suptitle("k-NN algorithm: sample green point is classified as red (k = " + str(k) + ")", fontsize=14,
                 fontweight='bold')


plt.show()

#%%
# Reconocimiento de digitos escritos a mano

def raw_pixels(img):
    return img.flatten()

def get_accuracy(predictions,labels):
    accuracy = (np.squeeze(predictions)==labels).mean()
    return accuracy * 100

def load_digits_and_labels(img):
    number_rows = img.shape[1] / 20
    rows = np.vsplit(img,img.shape[0]/20)
    
    digits = []
    for row in rows:
        row_cells = np.hsplit(row,number_rows)
        for digit in row_cells:
            digits.append(digit)
    digits = np.array(digits)
    
    labels = np.repeat(np.arange(10),len(digits)/10)
    return digits,labels

data = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/digits.png')
print("Forma de imagen: ",data.shape)
digits,labels = load_digits_and_labels(data)
rand = np.random.RandomState(1234)
shuffle = rand.permutation(len(digits))
digits,labels = digits[shuffle],labels[shuffle]

raw_descriptor = []
for img in digits:
    raw_descriptor.append(np.float32(raw_pixels(img)))


raw_descriptor = np.squeeze(raw_descriptor)
print("Shape raw_descriptor after squeeze: ",raw_descriptor.shape)

partition = int(0.8 * len(raw_descriptor))
raw_descriptor_train,raw_descriptor_test = np.split(raw_descriptor,[partition])

labels_train,labels_test = np.split(labels,[partition])


knn = cv2.ml.KNearest_create()
knn.train(raw_descriptor_train,cv2.ml.ROW_SAMPLE,labels_train)

results = defaultdict(list)
for k  in np.arange(1,10):
    ret,result,neighbours,dist = knn.findNearest(raw_descriptor_test,k)

    acc = get_accuracy(result,labels_test)
    print("Accuracy: {}".format("%.2f"%acc))
    results['80'].append(acc)
    
fig,ax = plt.subplots(1,1)
ax.set_xlim(0,10)
dim = np.arange(1,10)

for key in results:
    ax.plot(dim,results[key],linestyle='--',marker='o',label='80%')

plt.legend(loc='upper right',title='% training')
plt.title('Accuracy of the KNN model varying')
plt.ylabel('accuracy')
plt.xlabel('number of k')
plt.show()


#%%
# Variando el tamaño muestral

split_values = np.arange(0.1,1,0.1)
results = defaultdict(list)

for split in split_values:
    partition = int(split * len(raw_descriptor))
    raw_descriptor_train,raw_descriptor_test = np.split(raw_descriptor,[partition])
    labels_train,labels_test = np.split(labels,[partition])
    
    print('Training KNN model - raw pixels as features')
    knn.train(raw_descriptor_train,cv2.ml.ROW_SAMPLE,labels_train)
    for k in np.arange(1,10):
        ret,result,neighbours,dist = knn.findNearest(raw_descriptor_test,k)
        acc = get_accuracy(result,labels_test)
        print(" {}".format("%.2f"%acc))
        results[int(split * 100)].append(acc)
    
fig = plt.figure(figsize=(12,5))
plt.suptitle("K-NN handwritten digits recognition",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

ax = plt.subplot(1,1,1)
ax.set_xlim(0,10)
dim = np.arange(1,10)
for key in results:
    ax.plot(dim,results[key],linestyle='--',marker='o',label=str(key)+"%")

plt.legend(loc='upper right',title='% training')
plt.title('Accuracy of the KNN model varying both k and the percentage of images to train/test')
plt.xlabel('number of k')
plt.ylabel('accuracy')
plt.show()

#%%
# Haciendo un procesamiento previo
data = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/digits.png')


def deskew(img):
    image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1,skew,-0.5 * 20 * skew],[0,1,0]])
    img = cv2.warpAffine(img,M,(20,20),flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img



digits,labels = load_digits_and_labels(data)
rand = np.random.RandomState(1234)
shuffle = rand.permutation(len(digits))
digits,labels = digits[shuffle],labels[shuffle]

split_values = np.arange(0.1,1,0.1)
results = defaultdict(list)


raw_descriptor = []
for img in digits:
    raw_descriptor.append(np.float32(raw_pixels(deskew(img))))

for split in split_values:
    partition = int(split * len(raw_descriptor))
    raw_descriptor_train,raw_descriptor_test = np.split(raw_descriptor,[partition])
    labels_train,labels_test = np.split(labels,[partition])
    
    print('Training KNN model - raw pixels as features')
    knn.train(raw_descriptor_train,cv2.ml.ROW_SAMPLE,labels_train)
    for k in np.arange(1,10):
        ret,result,neighbours,dist = knn.findNearest(raw_descriptor_test,k)
        acc = get_accuracy(result,labels_test)
        print(" {}".format("%.2f"%acc))
        results[int(split * 100)].append(acc)
    
fig = plt.figure(figsize=(12,5))
plt.suptitle("K-NN handwritten digits recognition whit preprocesing",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

ax = plt.subplot(1,1,1)
ax.set_xlim(0,10)
dim = np.arange(1,10)
for key in results:
    ax.plot(dim,results[key],linestyle='--',marker='o',label=str(key)+"%")

plt.legend(loc='upper right',title='% training')
plt.title('Accuracy of the KNN model varying both k and the percentage of images to train/test')
plt.xlabel('number of k')
plt.ylabel('accuracy')
plt.show()

#%%
# Usnado descriptor HOG
# cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradient)
def get_hog():
    hog = cv2.HOGDescriptor((20,20),(8,8),(4,4),(8,8),9,1,-1,0,0.2,1,64,True)# blockSize tiene que ser multiplo de blockStride
    print("hog descriptor size: {}".format(hog.getDescriptorSize())) # Para cada imagen se muestra el tamaño de  descriptores de variables
    return hog
hog = get_hog()
hog_descriptor = []
for img in digits:
    hog_descriptor.append(hog.compute(deskew(img)))

hog_descriptor = np.squeeze(hog_descriptor)

results = defaultdict(list)
for split in split_values:
    partition = int(split * len(hog_descriptor))
    hog_descriptor_train,hog_descriptor_test = np.split(hog_descriptor,[partition])
    labels_train,labels_test = np.split(labels,[partition])
    
    print('Training KNN model - raw pixels as features')
    knn.train(hog_descriptor_train,cv2.ml.ROW_SAMPLE,labels_train)
    for k in np.arange(1,10):
        ret,result,neighbours,dist = knn.findNearest(hog_descriptor_test,k)
        acc = get_accuracy(result,labels_test)
        print(" {}".format("%.2f"%acc))
        results[int(split * 100)].append(acc)
        
fig = plt.figure(figsize=(12,5))
plt.suptitle("K-NN handwritten digits recognition whit preprocesing and HOG",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

ax = plt.subplot(1,1,1)
ax.set_xlim(0,10)
dim = np.arange(1,10)
for key in results:
    ax.plot(dim,results[key],linestyle='--',marker='o',label=str(key)+"%")

plt.legend(loc='upper right',title='% training')
plt.title('Accuracy of the KNN model varying both k and the percentage of images to train/test')
plt.xlabel('number of k')
plt.ylabel('accuracy')
plt.show()
#%%


