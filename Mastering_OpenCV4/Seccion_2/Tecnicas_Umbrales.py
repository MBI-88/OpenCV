"""La umbralizacion es simple, es todavia un metodo efectivo par particionar una imagen hacia
un nuevo plano. El objetivo de la segmentacion es modificar la representacion de una imagen 
hacia otra representacion que es mas facil de procesar."""
#%%
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import skimage as sk 

plt.rcParams['figure.figsize']=(12,8)
#%%
gray_image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/shades.png')
ret1,thresh1 = cv2.threshold(gray_image,50,255,cv2.THRESH_BINARY)# pixels con intesidad menor que 50 son negros
ret5,thresh5 = cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY)

plt.subplot(131)
plt.imshow(gray_image)
plt.title('Imagen gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(thresh1)
plt.title('Threshold image 1  (50)')
plt.axis('off')
plt.subplot(133)
plt.imshow(thresh5)
plt.title('Threshold image 5 (200)')
plt.axis('off')
plt.show()
#%%
def build_sample_image():
    tones = np.arange(start=50,stop=300,step=50)
    result = np.zeros(shape=(50,50,3),dtype='uint8')
    for  tone in tones:
        img = np.ones(shape=(50,50,3),dtype='uint8') * tone 
        result = np.concatenate((result,img),axis=1)
    
    return result

gray_image = build_sample_image()

ret1,thresh1 = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY)
ret2,thresh2 = cv2.threshold(gray_image,50,255,cv2.THRESH_BINARY)
ret3,thresh3 = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)
ret4,thresh4 = cv2.threshold(gray_image,150,255,cv2.THRESH_BINARY)

plt.subplot(221)
plt.imshow(thresh1)
plt.title('Threshold (0)')
plt.axis('off')
plt.subplot(222)
plt.imshow(thresh2)
plt.title('Threshold (50)')
plt.axis('off')
plt.subplot(223)
plt.imshow(thresh3)
plt.title('Threshold (100)')
plt.axis('off')
plt.subplot(224)
plt.imshow(thresh4)
plt.title('Threshold  (150)')
plt.axis('off')
plt.show()
#%%
# Umbralizacion simple

# Diferentes tipos de algoritmos 
# Nota: En los 2 ultimos metodos  la imagen debe tener un simple canal

"""cv2.THRESH_BINARY
   cv2.THRESH_BINARY_INV
   cv2.THRESH_TRUNC
   cv2.THRESH_TOZERO
   cv2.THRESH_TOZERO_INV
   cv2.THRESH_OTSU
   cv2.THRESH_TRIANGLE
   
Opreracion relizada por la funcion cv2.threshold() segun el tipo de algoritmo utilizado:
    se evalua pra el valor umbral (thresh) la intensidad de los pixeles de la imagen que 
    sobre pasen este valor son establesidos al valor maximo (maxval) (caso operacion binaria)
    (caso de operacion binaria invertida se establecen a 0)
    (caso de operacion truncada se establecen al valor de umbral (thresh))
    (caso de operacion cero se establecen al valor de la imagen)
    (caso de operacion cero invertida se establecen a 0)
    
En el caso de las operaciones OTSU,Triangle la operacion computa el valor umbral optimo ademas
del valor especificado como valor umbral"""

ret1,thresh1 = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)
ret2,thresh2 = cv2.threshold(gray_image,100,250,cv2.THRESH_BINARY)
ret3,thresh3 = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY_INV)
ret4,thresh4 = cv2.threshold(gray_image,100,250,cv2.THRESH_BINARY_INV)
ret5,thresh5 = cv2.threshold(gray_image,100,255,cv2.THRESH_TRUNC)
ret6,thresh6 = cv2.threshold(gray_image,100,255,cv2.THRESH_TOZERO)
ret7,thresh7 = cv2.threshold(gray_image,100,255,cv2.THRESH_TOZERO_INV)


plt.subplot(331)
plt.imshow(thresh1)
plt.title('Threshold Binaria maxval = 255')
plt.axis('off')
plt.subplot(332)
plt.imshow(thresh2)
plt.title('Threshold Binaria maxval = 250')
plt.axis('off')
plt.subplot(333)
plt.imshow(thresh3)
plt.title('Threshold Binaria invertida maxval = 255')
plt.axis('off')
plt.subplot(334)
plt.imshow(thresh4)
plt.title('Threshold  Binaria invertida maxval = 250')
plt.axis('off')
plt.subplot(335)
plt.imshow(thresh5)
plt.title('Threshold Truncada maxval = 255')
plt.axis('off')
plt.subplot(336)
plt.imshow(thresh6)
plt.title('Threshold Zero maxval = 255')
plt.axis('off')
plt.subplot(337)
plt.imshow(thresh7)
plt.title('Threshold Zeo invertida maxval = 255')
plt.axis('off')
plt.show()
#%%
# Aplicacion de tecnicas de umbralizacion a imagenes simples

image_real = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/sudoku.png')

ret1,thresh1 = cv2.threshold(image_real,60,255,cv2.THRESH_BINARY)
ret2,thresh2 = cv2.threshold(image_real,80,255,cv2.THRESH_BINARY)
ret3,thresh3 = cv2.threshold(image_real,100,255,cv2.THRESH_BINARY)
ret4,thresh4 = cv2.threshold(image_real,120,255,cv2.THRESH_BINARY)

plt.subplot(221)
plt.imshow(thresh1)
plt.title('Threshold (60)')
plt.axis('off')
plt.subplot(222)
plt.imshow(thresh2)
plt.title('Threshold (70)')
plt.axis('off')
plt.subplot(223)
plt.imshow(thresh3)
plt.title('Threshold (100)')
plt.axis('off')
plt.subplot(224)
plt.imshow(thresh4)
plt.title('Threshold  (130)')
plt.axis('off')
plt.show()
#%%
# Umbralizacion  adaptativa
""" Con la umbralizacion adaptativa se mejora la tecnica de umbralizacion ya que la anterior
depende de las diferentes condiciones de iluminacion de la imagen

La funcion utilizada en opencv es cv2.adaptativeThreshold() y sus parametros son:
    
    src: imaen fuente
    
    maxValue: establece el valor de los pixeles en la imagen de salida
    
    adaptativeMethod: establece los algoritmos a utilizar:
        
        Thresh_mean_C: El valor umbral T(x,y) es calculado como la media del blockeSize x blockeSize
        en el vecindario de (x,y) menos la constante C
        
        Thres_gaussian_C: El valor umbral T(x,y) es calculado como la suma de los pesos
        del blockSize x blockSize del vecindario de (x,y) menos la constante C
    
    blockSize: Establece el area del vecindario usado para calcular el valor de umbral para
    cada pixel y puede tomar valores 3,5,7,... y asi  sucesivamente.
    
    C: Es una constante sustraida a la media o la suma de los pesos medios. Comunmente este valor 
    es positivo pero puede ser  0 o (-)
    thresholdType: Establece el tipo de algoritmo de umbral"""

gray_image = cv2.cvtColor(image_real,cv2.COLOR_BGR2GRAY)
thresh_men_C2 = cv2.adaptiveThreshold(gray_image,250,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
thresh_mean_C3 = cv2.adaptiveThreshold(gray_image,250,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,3)
thresh_gaussian_C2 = cv2.adaptiveThreshold(gray_image,250,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
thresh_gaussian_C3 = cv2.adaptiveThreshold(gray_image,250,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,3)

plt.subplot(231)
plt.imshow(image_real)
plt.title('Image gray')
plt.axis('off')
plt.subplot(232)
plt.imshow(thresh_men_C2)
plt.title('Threshold adapt_mean block=11,c=2')
plt.axis('off')
plt.subplot(233)
plt.imshow(thresh_mean_C3)
plt.title('Threshold adapt_mean block=31,c=3')
plt.axis('off')
plt.subplot(234)
plt.imshow(thresh_gaussian_C2)
plt.title('Threshold adapt_gaussian block=11,c=2')
plt.axis('off')
plt.subplot(235)
plt.imshow(thresh_gaussian_C3)
plt.title('Threshold adapt_gaussian block=31,c=3')
plt.axis('off')
plt.show()

#%%

image_bilateral = cv2.bilateralFilter(gray_image,15,25,25)

thresh_men_C2 = cv2.adaptiveThreshold(image_bilateral,250,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
thresh_mean_C3 = cv2.adaptiveThreshold(image_bilateral,250,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,3)
thresh_gaussian_C2 = cv2.adaptiveThreshold(image_bilateral,250,cv2.ADAPTIVE_THRESH_GAUSSIAn_C,cv2.THRESH_BINARY,11,2)
thresh_gaussian_C3 = cv2.adaptiveThreshold(image_bilateral,250,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,3)

plt.subplot(231)
plt.imshow(image_real)
plt.title('Image gray')
plt.axis('off')
plt.subplot(232)
plt.imshow(thresh_men_C2)
plt.title('Threshold adapt_mean block=11,c=2')
plt.axis('off')
plt.subplot(233)
plt.imshow(thresh_mean_C3)
plt.title('Threshold adapt_mean block=31,c=3')
plt.axis('off')
plt.subplot(234)
plt.imshow(thresh_gaussian_C2)
plt.title('Threshold adapt_gaussian block=11,c=2')
plt.axis('off')
plt.subplot(235)
plt.imshow(thresh_gaussian_C3)
plt.title('Threshold adapt_gaussian block=31,c=3')
plt.axis('off')
plt.show()
#%%
# Algoritmo OTSU
"""
El algoritmo Otsu es una buena solucion cuando confrontamaos con imagenes bimodales.
Una imagen bimodal puede ser caracterizada por su histograma conteniendo  2 picos.
Ostsu automaticamente calcula el valor umbral  optimo  que separa ambos picos por la  
maximizacion de la variancia entre 2 calses de pixeles. El valor optimizado optimo  mini-
miza la varianza intra-calse.
El otsu es un algoritmo estatico poque utiliza informacion estatica derivada del histograma
(media,varianza,entropia).

El otsu se puede combinar con binrio,binario invertido,truncado,tozero y tozero invertido.
"""
image_leaf = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/leaf.png')
gray_leaf = cv2.cvtColor(image_leaf,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_leaf,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
histograma = cv2.calcHist([thresh],[0],None,[256],[0,255])

plt.subplot(221)
plt.imshow(image_leaf[:,:,::-1])
plt.title('Image')
plt.axis('off')
plt.subplot(222)
plt.imshow(gray_leaf)
plt.title('Gray image')
plt.axis('off')
plt.subplot(223)
plt.plot(histograma,color='r')
plt.axvline(x=ret,color='m',linestyle='--')
plt.title('Histogram')
plt.xlabel('Bins')
plt.ylabel('Frecuency')
plt.subplot(224)
plt.imshow(thresh)
plt.title('Otsu image')
plt.axis('off')
plt.show()

#%%
image_leaf_noise = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/leaf-noise.png')
gray_leaf_noise = cv2.cvtColor(image_leaf_noise,cv2.COLOR_BGRA2GRAY)

filter_gray_leaf = cv2.GaussianBlur(gray_leaf_noise,(15,15),0)
ret,thresh = cv2.threshold(gray_leaf_noise,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
histograma = cv2.calcHist([thresh],[0],None,[256],[0,256])
rt,thresh1 = cv2.threshold(filter_gray_leaf,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
histograma_f = cv2.calcHist([thresh1],[0],None,[256],[0,256])

plt.subplot(321)
plt.imshow(image_leaf_noise)
plt.title('Image noise')
plt.axis('off')
plt.subplot(322)
plt.imshow(gray_leaf_noise)
plt.title('Gray image noise')
plt.axis('off')
plt.subplot(323)
plt.plot(histograma,color='b')
plt.axvline(x=ret,color='m',linestyle='--')
plt.title('Histogram')
plt.xlabel('Bins')
plt.ylabel('Frecuency')
plt.subplot(324)
plt.imshow(thresh)
plt.title('Otsu thresh noise')
plt.axis('off')
plt.subplot(325)
plt.plot(histograma_f,color='r')
plt.axvline(x=rt,color='m',linestyle='--')
plt.xlabel('Bins')
plt.ylabel('Frecuency')
plt.subplot(326)
plt.imshow(thresh1)
plt.title('Otsu image filter')
plt.axis('off')
plt.show()
                                                

#%%
# Algoritmo tiangulo
"""
El algoritmo tiangulo es un metodo basdo en formas porque analiza la structura de el histograma
tratando de encontrar picos,planos entre otras variables del histograma.
Este algoritmo trabaja en 3 pasos :
    
    1) calcula una linea entre el maximo del histograma 
    en bmax sobre  la  axisa de nivel gris y el mas bajo  valor sobre la axisa de nivel gris.
    
    2) Se calcula la distancia  desde la linea (del primer  paso) hacia  el histograma para
    todos los valores de b [bmin -  bmax].
    
    3) Finalmente el nivel donde la distancia entre el histograma  y la linea es el maximao
    es  elegido como valor umbral.

"""
image_leaf_noise = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/leaf-noise.png')
gray_leaf_noise = cv2.cvtColor(image_leaf_noise,cv2.COLOR_BGRA2GRAY)

filter_gray_leaf = cv2.GaussianBlur(gray_leaf_noise,(15,15),0)
ret,thresh = cv2.threshold(gray_leaf_noise,0,255,cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
histograma = cv2.calcHist([thresh],[0],None,[256],[0,256])
rt,thresh1 = cv2.threshold(filter_gray_leaf,0,255,cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
histograma_f = cv2.calcHist([thresh1],[0],None,[256],[0,256])

plt.subplot(321)
plt.imshow(image_leaf_noise)
plt.title('Image noise')
plt.axis('off')
plt.subplot(322)
plt.imshow(gray_leaf_noise)
plt.title('Gray image noise')
plt.axis('off')
plt.subplot(323)
plt.plot(histograma,color='b')
plt.axvline(x=ret,color='m',linestyle='--')
plt.title('Histogram')
plt.xlabel('Bins')
plt.ylabel('Frecuency')
plt.subplot(324)
plt.imshow(thresh)
plt.title('Triangle thresh noise')
plt.axis('off')
plt.subplot(325)
plt.plot(histograma_f,color='r')
plt.axvline(x=rt,color='m',linestyle='--')
plt.xlabel('Bins')
plt.ylabel('Frecuency')
plt.subplot(326)
plt.imshow(thresh1)
plt.title('Triangle image filter')
plt.axis('off')
plt.show()
                      
#%%
# Umbralizacion en imagenes a colores

# Variante 1

image_lena = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/lenna.png')
ret2,thresh_lena = cv2.threshold(image_lena[:,:,::-1],150,255,cv2.THRESH_BINARY)
plt.subplot(121)
plt.imshow(image_lena[:,:,::-1])
plt.title('Lenna image')
plt.axis('off')
plt.subplot(122)
plt.imshow(thresh_lena)
plt.title('Image lenna thresh')
plt.axis('off')
plt.show()
#%%
# Variante 2

b,g,r = cv2.split(image_lena)
ret0,thresh0 = cv2.threshold(b,150,255,cv2.THRESH_BINARY)
ret1,thresh1 = cv2.threshold(g,150,255,cv2.THRESH_BINARY)
ret2,thresh2 = cv2.threshold(r,150,255,cv2.THRESH_BINARY)
image_merged = cv2.merge([thresh2,thresh1,thresh0])
plt.subplot(121)
plt.imshow(image_lena[:,:,::-1])
plt.title('Lenna image')
plt.axis('off')
plt.subplot(122)
plt.imshow(image_merged)
plt.title('Image lenna thresh')
plt.axis('off')
plt.show()




#%%
