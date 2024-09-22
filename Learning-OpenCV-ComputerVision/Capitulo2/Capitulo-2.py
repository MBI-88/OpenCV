# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:02:11 2021

@author: MBI
"""
#%%
import numpy as np
import cv2, os

#%%
# Leyendo y escribiendo archivos de imagenes

#%%
# Importando Imagenes de un HDD
"""
Modos soportados por cv2.imread():
    .cv2.IMREAD_COLOR: Esta es la opcion por defecto, provee una imagen BGR de 3 canales con valores de 8-bit de (0-255) para cada uno de los canales.
    
    .cv2.IMREAD_GRAYSCALE: Provee una imagen de 8-bits en escala de grices.
    
    .cv2.IMREAD_ANYCOLOR: Provee una imagen de 8-bit por canal o una imagen de 8-bit en escala de grices dependiendo de los metadatos del archivo.
    
    .cv2.IMREAD_UNCHANGED: Lee todos los datos de imagenes incluyendo los alpha o los canales transparente (si hay alguno ) como un 4 canal.
    
    .cv2.IMREAD_ANYDEPTH: Carga una imagen en escala de grices en su profundidad original.
    
    .cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR: Esta combinacion carga una imagen BGR en su profundidad de bit original.
    
    .cv2.IMREAD_REDUCED_GRAYSCALE_2: Carga una imagen en escala de grices  a la mitad de su resolucion original.
    
    .cv2.IMREAD_REDUCED_COLOR_2: Carga una imagen BGR a la mitad de su resolucion original.
    
    .cv2.IMREAD_REDUCED_GRAYSACLE_4: Carga una imagen en escala de grices a un cuarto de su resolucion original.
    
    .cv2.IMREAD_REDUCED_COLOR_4: Carga una imagen BGR a un cuarto de su resolucion original.
    
    .cv2.IMREAD_REDUCED_GRAY_8:  Carga una imagne en escala de grices a un octavo de su resolucion original.
    
    .cv2.IMREAD_REDUCED_COLOR_8: Carga una imagen BGR a un octavo de su resolucion original.

"""
#%%

grayimage = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/images/MyPic.png",
                       cv2.IMREAD_GRAYSCALE)

cv2.imshow("GRAY", grayimage)

while True:
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()

#%%
# Convirtiendo entre image y bytes

randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = np.array(randomByteArray)

grayimage = flatNumpyArray.reshape(300, 400)

cv2.imshow("RandomGRAY", grayimage)
while True:
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()

#%%

bgrimage = flatNumpyArray.reshape(100, 400, 3)

cv2.imshow("RandomBGR", bgrimage)
while True:
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
#%%
# Accediendo a los datos de una imagen con el metodo itemset que es mucho mas rapido que el tradicional indexado

img = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/images/MyPic.png")

img.itemset((150, 120, 0), 255)  # Establece el valor de un pixel en el canal azul.
print("Imagen cambiada: {} {}\n".format(img.item(150, 120, 0), img.shape))

cv2.imshow("Image arregalda", img)
while True:
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
#%%
img = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/images/MyPic.png")

img[:, :, 1] = 0

cv2.imshow("Image verde = 0", img)
while True:
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
#%%
# Utilizando regiones  de interes

img = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/images/MyPic.png")

my_roi = img[0:100, 0:100]

cv2.imshow("Image ROI", my_roi)
while True:
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()

#%%
# Lectura y escritura  de archivos de video

"""
Codes de video soportados:
    .0: Esta opcion es un archivo de video descomprimido. La extension del archivo debe ser .avi
    
    .cv2.VideoWriter_fourcc('I','4','2','0'): Esta opcion es un codificador descomprimido YUV de submuestreo chroma. Este codificador tiene gran compatibilidad pero produce  archivos muy grandes. La extension del archivo debe ser  .avi.
    
    .cv2.VideoWriter_fourcc('P','I','M','1'): Esta opcion es MPEG-1. La extension del archvivo debe ser .avi.
    
    .cv2.VideoWriter_fourcc('X','V','I','D'): Esta opcion es relativamente vieja MPEG-4. Es buena si se quiere limitar el tamaño del video resultante. La extension  del archivo debe ser .avi.
    
    .cv2.VideoWriter_fourcc('M','P','4','V'): Esta opcion es otra relativa a MPEG-4.
    
    .cv2.VideoWriter_fourcc('X','2','6','4'): Esta opcion es relativamente nueva MPEG-4. Talvez es la mejor opcion si se quiere limitar  el tamaño de el video resultante. La extension debe ser .mp4
    
    .cv2.VideoWriter_fourcc('T','H','E','O'): Esta opcion es Ogg Vorbis. La extension del archivo debe ser .ogv
    
    .cv2.VideoWriter_fourcc('F','L','V','1'): Esta opcion es un video de Flash. La extension del archivo debe ser .flv.
    

"""
#%%
# Transformado de una extension a otra.

videocapture = cv2.VideoCapture(
	"C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo2/MyInputVid.avi")

fps = videocapture.get(cv2.CAP_PROP_FPS)
size = (int(videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videowriter = cv2.VideoWriter(
	"C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo2/MyOutputVid.avi",
	cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

succes, frame = videocapture.read()
while succes:
	videowriter.write(frame)
	succes, frame = videocapture.read()

print("[*]Transformacion finalizada...")
#%%
# Capturando frames de una webcam

camaracapture = cv2.VideoCapture(0)
fps = 30  # se hace una estimacion para iniciar

size = (int(camaracapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camaracapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter(
	"C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo2/Mi_Video.avi",
	cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)

succes, frame = camaracapture.read()
numframeremaining = 10 * fps - 1
while succes and numframeremaining > 0:
	videoWriter.write(frame)
	succes, frame = camaracapture.read()
	numframeremaining -= 1

print("[*] Captura finalizada...")
#%%
# Mostrando frame de camaras en una ventana

clicked = False


def onMouse(event, x, y, flags, param):
	global clicked
	if event == cv2.EVENT_LBUTTONUP:
		clicked = True


cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('Mi_Ventana')
cv2.setMouseCallback('Mi_Ventana', onMouse)

print("Mostrando la entrada de camara. Click sobre la ventana  para salir")
ret, frame = cameraCapture.read()
while ret and cv2.waitKey(1) == -1 and not clicked:
	cv2.imshow('Mi_Ventana', frame)
	ret, frame = cameraCapture.read()

cv2.destroyWindow('Mi_Ventana')
cameraCapture.release()

#%%
"""
Parametros para la captura de entradas de teclado y mause:
    
    .cv2.EVENT_MOUSEMOVE: Este evento se refiere al movimineto del mause.
    .cv2.EVENT_LBUTTONDOWN: Este evento se refiere a presionar el click izquierdo.
    .cv2.EVENT_RBUTTONDOWN: Igual al anterior pero con el click derecho.
    .cv2.EVENT_MBUTTONDOWN: Se refiere a presionar la rudea del mouse.
    .cv2.EVENT_LBUTTONUP: Se refiere al click izqueirdo cuando se restablece su posicion.
    .cv2.EVENT_RBUTTONUP: Igual al anterior pero con el click derecho.
    .cv2.EVENT_MBUTTONUP: Igual al anterior pero con la rueda.
    .cv2.EVENT_LBUTTONDBLCLK: Este se refiere al doble click del boton izquierdo.
    .cv2.EVENT_RBUTTONDBLCLK: Igual al anterior pero con el boton derecho.
    .cv2.EVENT_MBUTTONDBLCLK: Igual al anterior pero con la rueda.
    
    .cv2.EVENT_FLAG_LBUTTON: Se refiere al boton izquierdo cuando esta comenzando a presionarse.
    .cv2.EVENT_FLAG_RBUTTON: Igual al anterior pero  con el derecho.
    .cv2.EVENT_FLAG_MBUTTON: Igual al anterior pero con la rueda.
    .cv2.EVENT_FLAG_CTRLKEY: Se refiere al boton Ctrl cuado se comienza a presionar.
    .cv2.EVENT_FLAG_SHIFTKEY: Igual al anterior pero con el boton Shift.
    .cv2.EVENT_FLAG_ALTKEY: Igual al anterior pero con le boton Alt.
 
"""