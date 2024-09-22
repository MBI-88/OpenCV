# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:27:26 2021

@author: MBI
"""
import cv2
import numpy as np

# %%
# Capturando frame desde la profundidad de una camara

"""
Metodos utilizados en la captura de profundidad de una camara:
    .cv2.CAP_OPENNI_DEPTH_MAP: Este es un mapa de profundidad  de una imagen en escala de grices  con cada valor de pixel estimado desde la camara a la superficie.
    .cv2.CAP_OPENNI_POINT_CLOUD_MAP: Este es un mapa de puntos en la nube de una imagen a color en la cual cada uno de los colores corresponde a x,y,z  dimensiones del espacio.
    .cv2.CAP_OPENNI_DISPARITY_MAP or cv2.CAP_OPENNI_DISPARITY_MAP_32F: Estos son  mapas disparejos de una imagen en escala de grices en la cual cada pixel  es lel esterio de disparidad de una  superficie.
    .cv2.CAP_OPENNI_DISPARITY_MAP_32F: Es un mapa de disparidad con 32 bit de valores flotantes.
    .cv2.CAP_OPENNI_BGR_IMAGE: Este es una imagen ordinaria BGR de una camara que captura la luz visible.
    .cv2.CAP_OPENNI_GRAY_IMAGE: Este es una imagen oridinaria de escala de grices de una camara que cptura la luz visible.
    .cv2.CAP_OPENNI_IR_IMAGE: Este es una imagen monocromatica de uan camara que captura la luz infra roja.
"""

# %%
"""
Parametros usados para el metodo  StereoSGM:
    
    .minDisparity: Minimo valor de disparidad posible. Normalmente es 0, pero a veces  algoritmos de rectificacion pueden aumentar la imagen por lo tanto el parametro necesita  ser ajustado.
    
    .numDisparities: Es la diferencia entere  la maxima disparidad y la minima disparidad. El valor es siempre mayor que 0. En la  implementacion actual este parametro  es dividido por 16.
    
    .blockSize: Ejecuta el tamaño del bloque. Debe ser el valor >= 1. Normalmente deberia  estar en el rango de 3 - 11.
    
    .P1: El primer parametro para controlar la suavidad de disparidad.
    
    .P2: El segundo parametro par controlar la suavidad de disparidad. El valor mas grande , la disparidad mas suavizada. P1 es la penalidad sobre el  cambio de disparidad por un excedente o un menos 1 entre pixeles vecinos. P2 es la penalidad sobre el cambio de disparidad por mas que 1 entre los pixeles vecinos. El algoritmo requiere  que P2 > P1.
    
    .disp12MaxDiff: Maxima diferencia permitida en el chequeo de  la disparidad de izquierda a derecha.
    
    .preFilterCap: Valor turncado para  los pixeles de imagen prefiltrados. El algoritmo primero computa la derivada de x en cada uno de los pixeles y corta sus valores por el intervalo [-prefilterCap,prefilterCap].
    
    .uniquenessRatio: Margen en porciento para el cual el mejor coste minimo  computado  por la funcion valor deberia ganar el segundo mejor valor para considerar encontrada la ejecucion correcta. Normalmente un valor esta entre 5-15 un rango suficiente bueno.
    
    .speckleWindowSize: Maximo tamaño de suavidad de la region de disparidad para considerar ruido speckles y invalidarla. Se establece a 0 para desabilitar el filtrado speckles. De otra manera se establece entre un rando de 50-200.
    
    .speckleRange: Maxima variacion de disparidad sin ninguno  de los componentes conectados. Si se hace el filtrado speckle, se obtiene el parametro para un valor positivo,el cual sera implicito en la multiplicacion por 16. Normalmente 1 o 2 es suficiente bueno.
    
    .mode: Se establece en el StereoSGM::MODE_HH para arrancar el el escalado completo, 2 pasos en el algoritmo de programacion dinamica. Consumira O(W*H*numDisparities) bytes para un tamaño de 640X480 stero y sera enorme para una imagen en HD.
    
"""
# %%
# Uso de la clase StereroSGM

minDisparity = 16
numDisparities = 192 - minDisparity
blockSize = 5
uniquenessRatio = 1
speckleWindowSize = 3
speckleRange = 3
disp12MaxDiff = 200
P1 = 600
P2 = 2400

stereo = cv2.StereoSGBM_create(minDisparity=minDisparity, numDisparities=numDisparities, blockSize=blockSize,
                               uniquenessRatio=uniquenessRatio, speckleRange=speckleRange,
                               speckleWindowSize=speckleWindowSize, disp12MaxDiff=disp12MaxDiff, P1=P1, P2=P2)

imgl = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo4/Frente.jpg")

imgr = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo4/Lateral.jpg")


def update(slidervalue=0):
	stereo.setBlockSize(cv2.getTrackbarPos('blockSize', 'Disparity'))
	stereo.setUniquenessRatio(cv2.getTrackbarPos("uniguenessRatio", "Disparity"))
	stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize', 'Disparity'))
	stereo.setDisp12MaxDiff(cv2.getTrackbarPos("disp12MaxDiff", "Disparity"))
	stereo.setSpeckleRange(cv2.getTrackbarPos("speckleRange", "Disparity"))
	
	disparity = stereo.compute(imgl, imgr).astype(np.float32) / 16.0
	
	cv2.imshow("Left", imgl)
	cv2.imshow("Right", imgr)
	cv2.imshow("Disparity", (disparity - minDisparity) / numDisparities)


cv2.namedWindow("Disparity")
cv2.createTrackbar('blockSize', 'Disparity', blockSize, 21, update)
cv2.createTrackbar('uniquenessRatio', 'Disparity', uniquenessRatio, 50, update)
cv2.createTrackbar('speckleWindowSize', 'Disparity', speckleWindowSize, 200, update)
cv2.createTrackbar('speckleRange', 'Disparity', speckleRange, 50, update)
cv2.createTrackbar('disp12MaxDiff', 'Disparity', disp12MaxDiff, 250, update)

update()
while True:
	if cv2.waitKey(-1) & 0xff == ord('q'):
		break

cv2.destroyAllWindows()

# %%
# Deteccion de foregroound y background con el algoritmo GrabCut.

original = cv2.imread(
	"C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo3/Learning OpenCV 4 Computer Vision with Python 3_page364_image31.jpg")

img = original.copy()
mask = np.zeros(img.shape[:2], dtype=np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (100, 1, 421, 378)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
img = img * mask2[:, :, np.newaxis]

cv2.imshow("Original", original)
cv2.imshow("Grabcut", img)

while True:
	if cv2.waitKey(-1) & 0xff == ord('q'):
		break

cv2.destroyAllWindows()

# Nota: La segmentacion se puede mejorar usando mas iteraciones sobre la imagen.
# %%
# Segmentacion de imagen usando el algoritmo Watershed.

image = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo4/5_of_diamonds.png")
gray_iamge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_iamge, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
print("Salida de dist_transform: ", dist_transform, "\n")
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = sure_bg.astype(np.uint8)

unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
print("Esto es markers: ", markers)
markers += 1
markers[unknown == 255] = 0

markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]

cv2.imshow("Carta", image)

while True:
	if cv2.waitKey(-1) & 0xff == ord('q'):
		break

cv2.destroyAllWindows()

# %%
