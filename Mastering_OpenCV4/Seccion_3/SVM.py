# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 18:41:55 2021

@author: MBI
"""
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from collections import defaultdict
#%%
# Maquinas de vector soporte

"""
Maquinas de vector soporte es una tecnica de entrenamiento supervisado que construye un hyperplane
en un espacion de alta dimension por la mejor separacion de las muestras de entrenamiento acorde a 
las clases asignadas.
La implementacion de SVM en opencv esta basada en LIBVM (libreria para maquinas de vector soporte)

Definicion de la  funcion: cv2.ml.SVM_create()
Parametros: 
    svmType: Establece el tipo de SVM. Estos tipos estan en la LIBM
    SVM_C_SVC: Vector soporte C que puede ser usado para clasificacion de n-clases (n >= 2)
    NU_SVC: Vector soporte V para clasificacion
    ONE_CLASS: Distribucion de la estimacion sobre las clases SVM
    EPS_SVR: Vector soporte E para regresion
    kernelType: Establese el tipo de kernel de la SVM. Posibles valores son:
        LINEAR,POLY,RBF (Radial Basis Function; buena eleccion en casi todos los casos),SIGMOID,CH2 (similar al
        al RBF),INTER (Kernel interseccion histograma , muy rapido).
    
    degre: Grado del kernel polinomial.
    gamma: Parametro del kernel  de las funciones (POLY,RBF,SIGMOID,CHI2)
    coef0: Parametro del kernel  la las funiones (POLY,SIGMOID)
    Cvalue: Parametro de una SVM para problemas de optimizacion (C_SVC,EPS_SVR,NU_SVR).
    nu: Parametro Para la SVM problemas de optimizacion (NU_SVC,ONE_CLASS,NU_SVR)
    p: Parametro para SVM problemas de optimizacion (EPS_SVR)
    classWeights: Pesos opcionales en el problema C_SVC , asignado a clases particulares.
    termCrit: Criterio de terminacion de el proceso iterativo en SVM.

El contructor por defecto inicializa los valores como sigue:
    svmType: C_SVC, kernelType: RBF, degree: 0, gamma: 1, coef0: 0, C: 1, nu:
    0, p: 0, classWeights: 0, termCrit: TermCriteria(MAX_ITER+EPS, 1000,FLT_EPSILON

Nota:
   La seleccion de la funcion del Kernel depende del dataset en cuestion. En esta medida el kernel RBF es 
   considerado en general una buena eleccion por la no linealidad de mapas de muestras que realiza hacia el 
   hiperplano del espacio dimensional .                                                  
"""
#%%
# Entendiendo SVM

labels = np.array([1,1,-1,-1,-1])
data = np.matrix([[500,10],[550,100],[300,10],[500,300],[10,600]],dtype=np.float32)

def svm_init(C=12.5,gamma=0.50625):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,100,1e-6))
    return model

svm_model = svm_init()

def svm_train(model,samples,resposes):
    model.train(samples,cv2.ml.ROW_SAMPLE,resposes)
    return model 

image = np.zeros((640,640,3),dtype='uint8')

def svm_predict(model,sample):
    return model.predict(sample)[1]

def show_svm_respose(model,image):
    colors = {1:(255,255,0),-1:(0,255,255)}
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sample = np.matrix([[j,i]],dtype=np.float32)
            response = svm_predict(model,sample)
            image[i,j] = colors[response.item(0)]
    
    cv2.circle(image,(500,10),10,(255,0,0),-1)
    cv2.circle(image,(550,100),10,(255,0,0),-1)
    
    cv2.circle(image,(300,10),10,(0,255,0),-1)
    cv2.circle(image,(500,300),10,(0,255,0),-1)
    cv2.circle(image,(10,600),10,(0,255,0),-1)
    
    support_vector = model.getUncompressedSupportVectors()
    for i in range(support_vector.shape[0]):
        cv2.circle(image,(support_vector[i,0],support_vector[i,1]),15,(0,0,255),6)


modelo = svm_train(svm_model,data,labels)
show_svm_respose(modelo,image)

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('silver')
plt.suptitle('Representation  de SVM',fontsize=14,fontweight='bold')
plt.imshow(image[:,:,::-1])
plt.title('Prediction SVM')
plt.axis('off')
plt.show()

#%%
# Reconocimiiento de digitos

image = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/digits.png',0)

def svm_evaluate(model,train,test):
    prediction = model.predict(train)[1].ravel()
    accuracy = (test == prediction).mean()
    return accuracy * 100

def svm_Init(C=12.5,gamma=0.50625):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,100,1e-6))
    return model

def deskew(img):
    #image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1,skew,-0.5 * 20 * skew],[0,1,0]])
    img = cv2.warpAffine(img,M,(20,20),flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

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

def get_hog():
    hog = cv2.HOGDescriptor((20,20),(8,8),(4,4),(8,8),9,1,-1,0,0.2,1,64,True)# blockSize tiene que ser multiplo de blockStride
    print("hog descriptor size: {}".format(hog.getDescriptorSize())) # Para cada imagen se muestra el tamaño de  descriptores de variables
    return hog

digits,labels = load_digits_and_labels(image)
rand = np.random.RandomState(1234)
shuffle = rand.permutation(len(digits))
digits,labels = digits[shuffle],labels[shuffle]

hog = get_hog()
hog_descriptors = []
for img in digits:
    hog_descriptors.append(hog.compute(deskew(img)))
hog_descriptors = np.squeeze(hog_descriptors)

partition = int(0.5 * len(hog_descriptors))
hog_d_train,hog_d_test = np.split(hog_descriptors,[partition])
labels_train,labels_test = np.split(labels,[partition])

model = svm_Init()
print("Train the model...")
response = svm_train(model,hog_d_train,labels_train)
accuracy = svm_evaluate(response,hog_d_test,labels_test)

print("Accuracy with 50% for training: ",accuracy)

#%%
# Variando C y gamma

partition = int(0.9 * len(hog_descriptors))
hog_d_train,hog_d_test = np.split(hog_descriptors,[partition])
labels_train,labels_test = np.split(labels,[partition])

results = defaultdict(list)
for C in [1,10,100,1000]:
    for gamma in [0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5]:
        model = svm_Init(C,gamma)
        model_trained = svm_train(model,hog_d_train,labels_train)
        acc = svm_evaluate(model_trained,hog_d_test,labels_test)
        print("Accuracy: {}".format("%.2f"%acc))
        results[C].append(acc)

fig = plt.figure(figsize=(10,6))
fig.patch.set_facecolor('yellow')
plt.suptitle('SVM  handwritten digits recognition',fontsize=14,fontweight='bold')
ax = plt.subplot(111)
ax.set_xlim(0,1.5)
dim = [0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5]

for key in results:
    ax.plot(dim,results[key],linestyle='--',marker='o',label=str(key))

plt.legend(loc='upper right',title='C')
plt.title("Accuracy of the SVM model varying both C and gamma")
plt.xlabel('gamma')
plt.ylabel('accuracy')
plt.show()

#%%
