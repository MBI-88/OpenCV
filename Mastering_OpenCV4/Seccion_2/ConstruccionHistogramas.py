#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize']=(15,15)

# %%
# Construccion y elabolracion de histogramas

def build_sample_image():
    "Build a sample image with 50x50 regions of different tones of gray"
    tones = np.arange(start=60,stop=240,step=30)
    result = np.ones(shape=(50,50,3),dtype='uint8') * 30
    
    #Build the image concatenating horizontally the regions
    for tone in tones:
        imag=np.ones(shape=(50,50,3),dtype='uint8') * tone
        result = np.concatenate((result,imag),axis=1)
    return result

def build_sample_image_2():
    "Build a sample image with 50x50 regions of different tones of gray flipping the ouput of build_sample_image()"
    img = np.fliplr(build_sample_image())
    return img

plt.subplot(121)
plt.imshow(build_sample_image())
plt.axis('off')
plt.subplot(122)
plt.imshow(build_sample_image_2())
plt.axis('off')
plt.show()
#%%
# Histogramas en escala de grices
"""
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[,
accumulate]])

To this, the following applies:
    
images: It represents the source image of type uint8 or float32
provided as a list (example, [gray_img]).

channels: It represents the index of the channel for which we
calculate the histogram provided as a list (for example, [0] for
grayscale images, or [0],[1],[2] for multi-channel images to
calculate the histogram for the first, second, or third channel,
respectively).

mask: It represents a mask image to calculate the histogram of a
specific region of the image defined by the mask. If this
parameter is equal to None, the histogram will be calculated with
no mask and the full image will be used.

histSize: It represents the number of bins provided as a list (for
example, [256]).

ranges: It represents the range of intensity values we want to
measure (for example, [0,256]).
"""

# Histogramas sin mascara de imagen

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/lenna.png')
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray_image],[0],None,[256],[0,256])

plt.plot(hist,color='r')
plt.show()

#%%
"""Si el promedio de valor de tono en una imagen es alto ejemplo 220 esto significa que
la mayor cantidad de pixeles de la imagen seran mas cercanos al color blanco.Por el contra-
rio si el promedio e pixeles es bajo ejemplo 30 significa que la mayor cantidad de pixeles
de la imagen sera cercana al negro"""

# Usando operaciones de adicion/sustraccion en la imagen

M = np.ones(shape=gray_image.shape,dtype='uint8') * 35
low_gray = cv2.subtract(gray_image,M)
hight_gray = cv2.add(gray_image,M)
hist_low_gray = cv2.calcHist([low_gray],[0],None,[256],[0,256])
hist_hight_gray = cv2.calcHist([hight_gray],[0],None,[256],[0,256])

plt.subplot(231)
plt.imshow(gray_image)
plt.title('Gray Image')
plt.axis('off')
plt.subplot(232)
plt.imshow(low_gray)
plt.title('Low Gray Image')
plt.axis('off')
plt.subplot(233)
plt.imshow(hight_gray)
plt.title('Hight Gray Image')
plt.axis('off')
plt.subplot(234)
plt.plot(hist,color='b')
plt.xlabel('bins')
plt.ylabel('Number of pixels')
plt.subplot(235)
plt.plot(hist_low_gray,color='g')
plt.xlabel('bins')
plt.ylabel('Number of pixels')
plt.subplot(236)
plt.plot(hist_hight_gray,color='y')
plt.xlabel('bins')
plt.ylabel('Number of pixels')
plt.show()

#%%
# Histogramas en escala de grices con mascara

"""Para aplicar una mascara a la imagen cargada y poder entonces calcular el histograma,
la mascara debe tener la misma forma que la imagne cargada y consiste en una imagen en 
negro (0) con la region seleccionada en blanco"""

mask = np.zeros(gray_image.shape[:2],dtype='uint8')
mask[30:190,30:190] = 255

hist_mask = cv2.calcHist([gray_image],[0],mask,[256],[0,256])

new_gray_imag=gray_image[30:190,30:190]

plt.subplot(221)
plt.imshow(gray_image)
plt.title('Gray Image')
plt.axis('off')
plt.subplot(222)
plt.plot(hist,color='r')
plt.xlabel('bins')
plt.ylabel('Number of pixels')
plt.subplot(223)
plt.imshow(new_gray_imag)
plt.title('Mask image')
plt.axis('off')
plt.subplot(224)
plt.plot(hist_mask,color='b')
plt.xlabel('bins')
plt.ylabel('Number of pixels')
plt.show()

#%%
# Histogramas de colors.Calculando el histograma en cada canal de una imagen BGR

def hist_color_image(imag):
    hist=[]
    for i in range(3):
        hist.append(cv2.calcHist([imag],[i],None,[256],[0,255]))
    return hist

image_bgr = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/lenna.png',cv2.IMREAD_COLOR)

hist_imagebgr = hist_color_image(image_bgr)
IMAGE = np.ones(shape=image_bgr.shape,dtype='uint8') * 15
add_image = cv2.add(image_bgr,IMAGE)
hist_addimage = hist_color_image(add_image)
subtrac_image = cv2.subtract(image_bgr,IMAGE)
hist_subtracimage = hist_color_image(subtrac_image)

plt.subplot(231)
plt.imshow(image_bgr[:,:,::-1])
plt.title('Image BGR')
plt.axis('off')
plt.subplot(232)
plt.imshow(add_image[:,:,::-1])
plt.title('Image Add')
plt.axis('off')
plt.subplot(233)
plt.imshow(subtrac_image[:,:,::-1])
plt.title('Image Susbtrac')
plt.axis('off')
plt.subplot(234)
plt.plot(hist_imagebgr[0],color='b')
plt.plot(hist_imagebgr[1],color='g')
plt.plot(hist_imagebgr[2],color='r')
plt.xlabel('bins')
plt.ylabel('Number of pixels')
plt.subplot(235)
plt.plot(hist_addimage[0],color='b')
plt.plot(hist_addimage[1],color='g')
plt.plot(hist_addimage[2],color='r')
plt.xlabel('bins')
plt.ylabel('Number of pixels')
plt.subplot(236)
plt.plot(hist_subtracimage[0],color='b')
plt.plot(hist_subtracimage[1],color='g')
plt.plot(hist_subtracimage[2],color='r')
plt.xlabel('bins')
plt.ylabel('Number of pixels')
plt.show()


#%%
# Personalizando visualizacion de histogramas

def plot_hist(hist_itmes,color):
    "Plots the histogram of a image"
    offset_down = 10
    offset_up = 10
    canvas = np.ones(shape=(300,256,3),dtype='uint8') * 255
    x_values = np.arange(256).reshape([256,1])
    
    for hist_item,col  in zip(hist_itmes,color):
        cv2.normalize(hist_item,hist_item,0+offset_down,300-offset_up,cv2.NORM_MINMAX)
        around = np.around(hist_item)
        hist = np.int32(around)
        pts = np.column_stack((x_values,hist))
        cv2.polylines(canvas,[pts],False,col,2)
        cv2.rectangle(canvas,(0,0),(255,298),(0,0,0),1)
    
    res = np.flipud(canvas)
    return res

res_im = plot_hist(hist_imagebgr,[(255,0,0),(0,255,0),(0,0,255)])
plt.imshow(res_im)
plt.title('Histograma personalizado')
plt.show()
#%%
# Comparando OpenCv,Numpy y Matplolib en histogramas
from timeit import default_timer as timer

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/lenna.png')
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Calculando el rendimiento de los histogramas de las diferentes funciones

start = timer()
hist_cv2 = cv2.calcHist([gray_image],[0],None,[256],[0,255])
end = timer()
exec_time_cv_hist = (end - start) * 1000
print('Tiempo tomado por openCv: ',exec_time_cv_hist)
print('\n')

start = timer()
hist_np = np.histogram(gray_image.ravel(),256,[0,255])
end = timer()
exec_time_np_hist = (end - start) * 1000
print('Tiempo tomado por numpy: ',exec_time_np_hist)
print('\n')

start = timer()
(n,bins,patches) = plt.hist(gray_image.ravel(),256,[0,255])
end = timer()
exec_time_plt_hist = (end - start) * 1000
print('Tiempo tomado por matpotlib: ',exec_time_plt_hist)


#%%
# Ecualizacion de histogramas
# Histogramas en escala de grices

gray_image_eq = cv2.equalizeHist(gray_image)
hist_original = cv2.calcHist([gray_image],[0],None,[256],[0,255])
hist_eq = cv2.calcHist([gray_image_eq],[0],None,[256],[0,255])

plt.subplot(221)
plt.imshow(gray_image)
plt.title('Original')
plt.axis('off')
plt.subplot(222)
plt.plot(hist_original,color='b')
plt.ylabel('Frecuency')
plt.xlabel('Bins')
plt.subplot(223)
plt.imshow(gray_image_eq)
plt.title('Equalizada')
plt.axis('off')
plt.subplot(224)
plt.plot(hist_eq,color='r')
plt.ylabel('Frecuency')
plt.xlabel('Bins')
plt.show()
#%%
# Histogramas equalizados en colores

def equalize_hist_color(img):
    "Equalize the image splitting the image applying cv2.equalizeHist() to each channel and merging result"
    channel = cv2.split(img)
    eq_channels = []
    for ch in channel:
        eq_channels.append(cv2.equalizeHist(ch))
    
    eq_image = cv2.merge(eq_channels)
    
    return  eq_image

hist_original_image = []
for i in range(3):
    hist_original_image.append(cv2.calcHist([image],[i],None,[256],[0,255]))

imag_equalized = equalize_hist_color(image)

hist_equalized_imag = []
for i in  range(3):
    hist_equalized_imag.append(cv2.calcHist([imag_equalized],[i],None,[256],[0,255]))


plt.subplot(221)
plt.imshow(image[:,:,::-1])
plt.title('Original')
plt.axis('off')
plt.subplot(222)
plt.plot(hist_original_image[0],color='b')
plt.plot(hist_original_image[1],color='g')
plt.plot(hist_original_image[2],color='r')
plt.xlabel('bins')
plt.ylabel('Number of pixel')
plt.subplot(223)
plt.imshow(imag_equalized[:,:,::-1])
plt.title('Equalizada')
plt.axis('off')
plt.subplot(224)
plt.plot(hist_equalized_imag[0],color='b')
plt.plot(hist_equalized_imag[1],color='g')
plt.plot(hist_equalized_imag[2],color='r')
plt.xlabel('bins')
plt.ylabel('Number of pixel')
plt.show()

#%%
"""Para mejorar el cambio dramatico que trae consigo el BGR lo mejor es cambiar a un 
espacio de color que contenga intesidad de luminancia (Yuv,Lab,HSV y HSL)
Se aplica la ecualizacion de histograma solamente a un canal de luminancia y despues se 
hace la transformacion a BGR"""

def equalize_hist_color_hsv(img):
    "Ecualize the image spliting after HSV convertion and applying cv2.equalizeHist()"
    
    H,S,V = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H,S,eq_V]),cv2.COLOR_HSV2BGR)
    return eq_image

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/lenna.png')
hist_imag = []
for i in  range(3):
    hist_imag.append(cv2.calcHist([image],[i],None,[256],[0,255]))

imag_hsv_eq = equalize_hist_color_hsv(image)

hist_imag_hsv = []
for i in range(3):
    hist_imag_hsv.append(cv2.calcHist([imag_hsv_eq],[i],None,[256],[0,255]))

plt.subplot(221)
plt.imshow(image[:,:,::-1])
plt.title('Original')
plt.axis('off')
plt.subplot(222)
plt.plot(hist_imag[0],color='b')
plt.plot(hist_imag[1],color='g')
plt.plot(hist_imag[2],color='r')
plt.xlabel('bins')
plt.ylabel('Number of pixel')
plt.subplot(223)
plt.imshow(imag_hsv_eq[:,:,::-1])
plt.title('Equalizada HSV')
plt.axis('off')
plt.subplot(224)
plt.plot(hist_imag_hsv[0],color='b')
plt.plot(hist_imag_hsv[1],color='g')
plt.plot(hist_imag_hsv[2],color='r')
plt.xlabel('bins')
plt.ylabel('Number of pixel')
plt.show()

#%%
# Adaptacion de limite de contraste en ecualizacion histogramas (CLAHE)
# Este algoritmo se utiliza en en el mejoramiento del contraste de una imagen

image_bgr = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/lenna.png')
image_gray = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit = 2.0)
image_gray_clahe = clahe.apply(image_gray)

plt.subplot(121)
plt.imshow(image_gray)
plt.title('Image gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(image_gray_clahe)
plt.title('Image clahe')
plt.axis('off')
plt.show()


#%%

"""El algoritmo CLAHE da un mejor rendimiento y resultado que la equalizacion de histograma
en muchas situaciones. Es comun utilizar como primer paso en muchas aplicaciones de vision
computacional (ejemplo procesacmiento de caras entre otros """

def equalize_clahe_color_hsv(img):
    
    cla = cv2.createCLAHE(clipLimit = 4.0)
    H,S,V = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2HSV))
    eq_V = cla.apply(V)
    eq_image = cv2.cvtColor(cv2.merge([H,S,eq_V]),cv2.COLOR_HSV2BGR)
    return eq_image

def equalize_clahe_color_lab(img):
    cla = cv2.createCLAHE(clipLimit = 4.0)
    L,a,b = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2Lab))
    eq_L = cla.apply(L)
    eq_image = cv2.cvtColor(cv2.merge([eq_L,a,b]),cv2.COLOR_Lab2RGB)
    return eq_image

def equalize_clahe_color_yuv(img):
    cla = cv2.createCLAHE(clipLimit = 4.0)
    Y,U,V = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2YUV))
    eq_Y = cla.apply(Y)
    eq_image = cv2.cvtColor(cv2.merge([eq_Y,U,V]),cv2.COLOR_YUV2BGR)
    return eq_image

def  equalize_clahe_color(img):
    cla = cv2.createCLAHE(clipLimit = 4.0)
    channel = cv2.split(img)
    eq_channel = []
    for ch in channel:
        eq_channel.append(cla.apply(ch))
    
    eq_imag = cv2.merge(eq_channel[::-1])
    return eq_imag


eq_hsv = equalize_clahe_color_hsv(image)
eq_lab = equalize_clahe_color_lab(image)
eq_yuv = equalize_clahe_color_yuv(image)
eq_color = equalize_clahe_color(image)

plt.subplot(221)
plt.imshow(eq_hsv[:,:,::-1])
plt.title('Image hsv')
plt.axis('off')
plt.subplot(222)
plt.imshow(eq_lab)
plt.title('Image lab')
plt.axis('off')
plt.subplot(223)
plt.imshow(eq_yuv[:,:,::-1])
plt.title('Image yuv')
plt.axis('off')
plt.subplot(224)
plt.imshow(eq_color)
plt.title('Image color')
plt.axis('off')
plt.show()
#%%        
# Comparacion de Histogramas

"""Un acercamiento comun para comparar imagenes es dividir la imagen en ciertos numeros
de regiones (comunentedel mismo tamaño), calcular el histograma para cada region y finalmente
concatenar  todos los histogramas para crear una representacion de la imagen. """

image_bgr = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/lenna.png')
image_gray = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)

Matrix = np.ones(image_gray.shape,dtype='uint8') *  35
image_add = cv2.add(image_gray,Matrix)
image_sub = cv2.subtract(image_gray,Matrix)
image_blur = cv2.blur(image_gray,(10,10))


hist_image_gray = cv2.calcHist([image_gray],[0],None,[256],[0,255])
hist_image_add = cv2.calcHist([image_add],[0],None,[256],[0,255])
hist_image_sub = cv2.calcHist([image_sub],[0],None,[256],[0,255])
hist_image_blur = cv2.calcHist([image_blur],[0],None,[256],[0,255])

comp_add_cor = cv2.compareHist(hist_image_gray,hist_image_add,method=cv2.HISTCMP_CORREL)
comp_add_chi = cv2.compareHist(hist_image_gray,hist_image_add,method=cv2.HISTCMP_CHISQR)
comp_add_bha = cv2.compareHist(hist_image_gray,hist_image_add,method=cv2.HISTCMP_BHATTACHARYYA)
comp_add_int = cv2.compareHist(hist_image_gray,hist_image_add,method=cv2.HISTCMP_INTERSECT)

comp_sub_cor = cv2.compareHist(hist_image_gray,hist_image_sub,method=cv2.HISTCMP_CORREL)
comp_sub_chi = cv2.compareHist(hist_image_gray,hist_image_sub,method=cv2.HISTCMP_CHISQR)
comp_sub_bha = cv2.compareHist(hist_image_gray,hist_image_sub,method=cv2.HISTCMP_BHATTACHARYYA)
comp_sub_int = cv2.compareHist(hist_image_gray,hist_image_sub,method=cv2.HISTCMP_INTERSECT)

comp_blur_cor = cv2.compareHist(hist_image_gray,hist_image_blur,method=cv2.HISTCMP_CORREL)
comp_blur_chi = cv2.compareHist(hist_image_gray,hist_image_blur,method=cv2.HISTCMP_CHISQR)
comp_blur_bha = cv2.compareHist(hist_image_gray,hist_image_blur,method=cv2.HISTCMP_BHATTACHARYYA)
comp_blur_int = cv2.compareHist(hist_image_gray,hist_image_blur,method=cv2.HISTCMP_INTERSECT)


plt.subplot(441)
plt.imshow(image_gray)
plt.title('Imagen gray')
plt.axis('off')
plt.subplot(442)
plt.imshow(image_add)
plt.title('Image add correl: {}'.format(comp_add_cor))
plt.axis('off')
plt.subplot(443)
plt.imshow(image_sub)
plt.title('Image sub correl: {}'.format(comp_sub_cor))
plt.axis('off')
plt.subplot(444)
plt.imshow(image_blur)
plt.title('Image blur correl: {}'.format(comp_blur_cor))
plt.axis('off')

plt.subplot(445)
plt.imshow(image_gray)
plt.title('Imagen gray')
plt.axis('off')
plt.subplot(446)
plt.imshow(image_add)
plt.title('Image add chisqr: {}'.format(comp_add_chi))
plt.axis('off')
plt.subplot(447)
plt.imshow(image_sub)
plt.title('Image sub chisqr: {}'.format(comp_sub_chi))
plt.axis('off')
plt.subplot(448)
plt.imshow(image_blur)
plt.title('Image blur chisqr: {}'.format(comp_blur_chi))
plt.axis('off')

plt.subplot(449)
plt.imshow(image_gray)
plt.title('Imagen gray')
plt.axis('off')
plt.subplot(4,4,10)
plt.imshow(image_add)
plt.title('Image add intersect: {}'.format(comp_add_int))
plt.axis('off')
plt.subplot(4,4,11)
plt.imshow(image_sub)
plt.title('Image sub intersect: {}'.format(comp_sub_int))
plt.axis('off')
plt.subplot(4,4,12)
plt.imshow(image_blur)
plt.title('Image blur intersect: {}'.format(comp_blur_int))
plt.axis('off')

plt.subplot(4,4,13)
plt.imshow(image_gray)
plt.title('Imagen gray')
plt.axis('off')
plt.subplot(4,4,14)
plt.imshow(image_add)
plt.title('Image add bhat: {}'.format(comp_add_bha))
plt.axis('off')
plt.subplot(4,4,15)
plt.imshow(image_sub)
plt.title('Image sbub bhat: {}'.format(comp_sub_bha))
plt.axis('off')
plt.subplot(4,4,16)
plt.imshow(image_blur)
plt.title('Image blur bhat: {}'.format(comp_blur_bha))
plt.axis('off')

plt.show()
#%%





