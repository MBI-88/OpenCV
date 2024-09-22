# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:20:28 2021

@author: MBI
"""

import cv2
import numpy as np
#%%
# Sustraccion de fondos basicos.

BlUR_Radius = 21
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
cap = cv2.VideoCapture(0)

for i in range(10):
    success,frame = cap.read()
    
if not success:
    exit(1)
        
gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_background = cv2.GaussianBlur(gray_background, (BlUR_Radius,BlUR_Radius), 0)

success,frame = cap.read()

while success:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (BlUR_Radius,BlUR_Radius),0)
    
    diff = cv2.absdiff(gray_background, gray_frame)
    _,thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    cv2.erode(thresh, erode_kernel,thresh,iterations=2)
    cv2.dilate(thresh, dilate_kernel,thresh,iterations=2)
    
    contours,hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) > 400:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0),2)
    
    cv2.imshow("diff", diff)
    cv2.imshow("thresh", thresh)
    cv2.imshow("detection", frame)
    
    if cv2.waitKey(1) == 27:
        break

    success,frame = cap.read()

cv2.destroyAllWindows()
#%%
# Usando el algoritmo MOG
"""
El sustractor MOG devuelve tres mascaras: blanco (255) para segmentacion principal (foreground segments), gris (127) para segmentacion de sombras (shadow segments), negro (0) para segmentacion de fondo (background segments)
"""

bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True) # Nota: Si se pone False la deteccion pierde presicion
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))

cap = cv2.VideoCapture("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo8/hallway.mpg")
success,frame = cap.read()

while success:
    fg_mask = bg_subtractor.apply(frame)
    
    _,thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    cv2.erode(thresh,erode_kernel,thresh,iterations=2)
    cv2.dilate(thresh,dilate_kernel,thresh,iterations=2)
    
    contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) > 1000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
    
    cv2.imshow("mog", fg_mask)
    cv2.imshow("thresh", thresh)
    cv2.imshow("detection", frame)
    
    if cv2.waitKey(30) == 27:
        break

    success,frame = cap.read()

cv2.destroyAllWindows()
#%%
# Usando un  sustractor de fondo KNN

bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,11))

cap = cv2.VideoCapture("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo8/traffic.flv")

success,frame = cap.read()

while success:
    fg_mask = bg_subtractor.apply(frame)
    _,thresh = cv2.threshold(fg_mask, 230, 255, cv2.THRESH_BINARY)
    cv2.erode(thresh,erode_kernel,thresh,iterations=2)
    cv2.dilate(thresh, dilate_kernel,thresh,iterations=2)
    
    contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) > 1000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
    
    cv2.imshow("Car knn",fg_mask)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Detection", frame)
    
    if cv2.waitKey(30) == ord("q"):
        break
    success,frame = cap.read()

cv2.destroyAllWindows()
#%%
# Usando GMG sobre sustractores de fondo

"""
Mas sustractores de fondo son accequibles en el modulo cv2.bgsegm estos pueden ser creados usando las siguuientes funciones:
    
    cv2.bgsegm.createBackgroundSubtractorCNT
    cv2.bgsegm.createBackgroundSubtractorGMG
    cv2.bgsegm.createBackgroundSubtractorGSOC
    cv2.bgsegm.createBackgroundSubtractorLSBP
    cv2.bgsegm.createBackgroundSubtractorMOG
    cv2.bgsegm.createSyntheticSequenceGenerator

Nota: Estas funciones no soportan la deteccion de sombras como parametro y todas soportan el metodo apply()
"""

bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,9))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,11))

cap = cv2.VideoCapture("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo8/traffic.flv")
success,frame = cap.read()

while success:
    fg_mask = bg_subtractor.apply(frame)
    _,thresh = cv2.threshold(fg_mask,240,255,cv2.THRESH_BINARY)
    cv2.erode(thresh,erode_kernel,thresh,iterations=2)
    cv2.dilate(thresh, dilate_kernel,thresh,iterations=2)
    
    contours,hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for  c in contours:
        if cv2.contourArea(c) > 1000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
    
    cv2.imshow("GMG",fg_mask)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Detection", frame)
    
    if cv2.waitKey(30) == ord('q'):
        break
    success,frame = cap.read()

cv2.destroyAllWindows()
#%%
# Descripcion de la funcion historgram

"""
Definicion de cv2.calcHist():
    
    .images : Es una lista de una o mas imagenes. Todas deben tener la misma profundidad de color y el mismo tamaño.
    .channels  : Es una lista de indices de canales usados para computar  el histograma. Por ejemplo channel=[0] significa que solo el primer canal es usado
    .mask : Este parametro es  None por defecto , cada region de la imagen sera computada. Si no es none entonces debe ser un array de 8 bits con el mismo tamaño de la imagen en la imagenes.
    .histSize : Este parametro es una lista de numeros de bondaridades de histogramas usado para cada uno de los canales. La longitud de este debe ser igual  a la del channels. Ejemplo channel=[0] y histSize=[180].
    .ranges : Es una lista que especifica el rango de valores para usar  en cada canal. La longitud de ranges debe ser el doble de la del channels. Ejemplo channel=[0] , histSize =[180] y ranges=[0,180].
    .hist : Es opcional , es usado para direccionar la salida.
    .accumulate : Es opcional se utiliza para acumular los histogramas de las imagenes sin que se pierda el primero calculado, los valores que toma son booleano.


Definicion de cv2.calcBackProject():
    
    .images : Es una lista de imagenes, todas deben tener la misma profundidad.
    .channels : Es igual a cv2.calcHist().
    .hist : Es el histograma.
    .ranges : Es igual a cv2.calcHist()
    .scale : Este parametro es un factor de escala. La projeccion trasera multiplica por este factor .
    .dst : Este parametro direcciona la salida.
"""
#%%
# Implementacion de MeanShift

cap = cv2.VideoCapture(0)

for i in range(10):
    success,frame = cap.read()

if not success:
    exit(1)

frame_h,frame_w, = frame.shape[:2]
w = frame_w//8
h = frame_h//8
x = frame_w//2 - w//2
y = frame_h//2 - h//2
track_window = (x,y,w,h)
roi = frame[y:y+h,x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [2],None,[180],[0,180])
cv2.normalize(roi_hist, roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,10,1)

success,frame = cap.read()

while success:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_proj = cv2.calcBackProject([hsv],[2],roi_hist,[0,180],1)
    
    num_iters,track_window = cv2.meanShift(back_proj, track_window, term_crit)
    
    x,y,w,h = track_window
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,250),2)
    
    cv2.imshow("Back_prjection",back_proj)
    cv2.imshow("MeanShift",frame)
    
    if cv2.waitKey(30) == ord("q"):
        break
    succes,frame = cap.read()
    

cv2.destroyAllWindows()
#%%
# Usando un mejoramiento en MeanShift (CamShift)

cap = cv2.VideoCapture(0)

for i in range(10):
    success,frame = cap.read()

if not success:
    exit(1)

frame_h,frame_w, = frame.shape[:2]
w = frame_w//8
h = frame_h//8
x = frame_w//2 - w//2
y = frame_h//2 - h//2
track_window = (x,y,w,h)
roi = frame[y:y+h,x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0],None,[180],[0,180])
cv2.normalize(roi_hist, roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,10,1)

succes,frame = cap.read()

while succes:
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_proj = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
    rotated_rect,track_window = cv2.CamShift(back_proj, track_window, term_crit)
    
    box_points = cv2.boxPoints(rotated_rect)
    box_points = np.int0(box_points)
    cv2.polylines(frame, [box_points],True,(255,0,0),2)
    
    cv2.imshow("Back_projection", back_proj)
    cv2.imshow("Camshift", frame)
    
    if cv2.waitKey(1) == ord("q"):
        break
    succes,frame = cap.read()

cv2.destroyAllWindows()
#%%
# Usando el metodo Kalma filter para predecir posicion de un objeto

img = np.zeros((800,800,3),np.uint8)

kalma = cv2.KalmanFilter(4,2)
kalma.measurementMatrix = np.array(
    [[1,0,0,0],[0,1,0,0]],np.float32)

kalma.transitionMatrix = np.array(
    [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)

kalma.processNoiseCov = np.array(
    [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

last_measurement = None
last_prediction = None

def on_mouse_moved(event,x,y,flags,param):
    global img,kalman,last_measurement,last_prediction 
    
    measurement = np.array([[x],[y]],np.float32)
    if last_measurement is None:
        kalma.statePre = np.array([[x],[y],[0],[0]],np.float32)
        kalma.statePost = np.array([[x],[y],[0],[0]],np.float32)
        prediction = measurement
    else:
        kalma.correct(measurement)
        prediction = kalma.predict()
        cv2.line(img,(last_measurement[0],last_measurement[1]),(measurement[0],measurement[1]),(0,255,0))
        cv2.line(img,(last_prediction[0],last_prediction[1]),(prediction[0],prediction[1]),(0,0,255))

    last_prediction = prediction.copy()
    last_measurement = measurement


cv2.namedWindow("Kalma_tracker")
cv2.setMouseCallback("Kalma_tracker",on_mouse_moved)

while True:
    cv2.imshow("Kalma_tracker", img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

#%%
# Siguiendo peatones

class Peaton():
    def __init__(self,id,hsv_frame,track_window):
        self.id = id
        self.track_window = track_window
        self.term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,10,1)
        
        x,y,w,h = track_window
        roi = hsv_frame[y:y+h,x:x+w]
        roi_hist = cv2.calcHist([roi],[0],None,[16],[0,180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist,0,255,cv2.NORM_MINMAX)
        
        self.kalma = cv2.KalmanFilter(4,2)
        self.kalma.measurementMatrix = np.array(
    [[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalma.transitionMatrix = np.array(
    [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalma.processNoiseCov = np.array(
    [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
        
        cx = x + w/2
        cy = y + h/2
        self.kalma.statePre = np.array([[cx],[cy],[0],[0]],np.float32)
        self.kalma.statePost = np.array([[cx],[cy],[0],[0]],np.float32)
    
    def update(self,frame,hsv_frame):
        back_proj = cv2.calcBackProject([hsv_frame],[0],self.roi_hist,[0,180],1)
        ret,self.track_window = cv2.meanShift(back_proj, self.track_window,self.term_crit)
        x,y,w,h = self.track_window
        center = np.array([x+w/2,y+h/2],np.float32)
        
        prediction = self.kalma.predict()
        estimate = self.kalma.correct(center)
        center_offset = estimate[:,0][:2] - center
        self.track_window = (x + int(center_offset[0]),y+int(center_offset[1]),w,h)
        x,y,w,h = self.track_window
        
        cv2.circle(frame,(int(prediction[0]),int(prediction[1])),4,(255,0,0),-1)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0),2)
        cv2.putText(frame, "ID:%d"% self.id,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),1,cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo8/pedestrians.avi")
    
    bg_subtractor = cv2.createBackgroundSubtractorKNN()
    history_length = 20
    bg_subtractor.setHistory(history_length)
    
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,3))
    
    peaton = []
    num_history_frame_poputation = 0
    
    while True:
        grabbed ,frame = cap.read()
        if (grabbed is False):
            break
        fg_mask = bg_subtractor.apply(frame)
        if num_history_frame_poputation < history_length:
            num_history_frame_poputation += 1
            continue
        _,thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        
        cv2.erode(thresh,erode_kernel,thresh,iterations=2)
        cv2.dilate(thresh,dilate_kernel,thresh,iterations=2)
        
        contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        should_initialize_pedestrans = len(peaton) == 0
        id = 0
        for c in contours:
            if cv2.contourArea(c) > 500:
                (x,y,w,h) = cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                if should_initialize_pedestrans:
                    peaton.append(Peaton(id,hsv_frame,(x,y,w,h)))
            
            id += 1
        
        for pedestrians in peaton:
            pedestrians.update(frame, hsv_frame)
        
        cv2.imshow("Peatones", frame)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()
    cv2.destroyWindow("Peatones")
#%%
