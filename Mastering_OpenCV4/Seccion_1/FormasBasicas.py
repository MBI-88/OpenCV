#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255),
'yellow': (0, 255, 255), 'magenta': (255, 0, 255), 'cyan': (255, 255, 0),
'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (125, 125, 125),
'rand': np.random.randint(0, high=256, size=(3,)).tolist(), 'dark_gray':
(50, 50, 50), 'light_gray': (220, 220, 220)}


# In[3]:


image=np.zeros((500,500,3),dtype='uint8')
image[:]=colors['light_gray']
separation = 40
for key in colors:
    cv2.line(image,(0,separation),(500,separation),colors[key],10)
    separation += 40

plt.imshow(image)
plt.title('Diccionrio de colores')
plt.show() # Mostrando el diccionario de colores que se utilizara


# In[4]:


# Dibujando formas
image=np.zeros((400,400,3),dtype='uint8')
image[:]=colors['light_gray']
img=cv2.line(image,(0,0),(400,400),colors['green'],3)
img=cv2.line(image,(0,400),(400,0),colors['red'],10)
img=cv2.line(image,(200,0),(200,400),colors['blue'],3)
img=cv2.line(image,(0,200),(400,200),colors['yellow'],10)

plt.imshow(img)
plt.title('Dibujando lineas')
plt.show()


# In[5]:


image=np.zeros((400,400,3),dtype='uint8')
image[:]=colors['light_gray']
rectangulo=cv2.rectangle(image,(10,50),(60,300),colors['green'],3)
rectangulo=cv2.rectangle(image,(80,50),(130,300),colors['blue'],-1)
rectangulo=cv2.rectangle(image,(150,50),(350,100),colors['red'],-1)
rectangulo=cv2.rectangle(image,(150,150),(350,300),colors['cyan'],10)
plt.imshow(rectangulo)
plt.title('Dibujando rectangulos')
plt.show()


# In[6]:


image=np.zeros((400,400,3),dtype='uint8')
image[:]=colors['light_gray']
circulos=cv2.circle(image,(50,50),20,colors['blue'],3)
circulos=cv2.circle(image,(100,100),30,colors['red'],-1)
circulos=cv2.circle(image,(200,200),40,colors['yellow'],10)
circulos=cv2.circle(image,(300,300),40,colors['green'],-1)
plt.imshow(circulos)
plt.title('Dibujando circulos')
plt.show()


# In[7]:


# Dibujando formas avanzadas
image=np.zeros((300,300,3),dtype='uint8')
image[:]=colors['light_gray']
# Clip line
cv2.line(image,(0,0),(300,300),colors['green'],3)
cv2.rectangle(image,(0,0),(100,100),colors['red'],3)
ret,p1,p2=cv2.clipLine((0,0,100,100),(0,0),(300,300)) # ret es true  si almenos uno de los puntos esta dentro del rectangulo 
if ret:
    cv2.line(image,p1,p2,colors['yellow'],3)
    plt.imshow(image)
    plt.title('Dibujando lineas cortadas')
    plt.show()
else: print('Los puntos estan fuera del rectangulo')


# In[8]:


image=np.zeros((300,300,3),dtype='uint8')
image[:]=colors['light_gray']
cv2.arrowedLine(image,(50,50),(200,50),colors['red'],3,8,0,0.1)
cv2.arrowedLine(image,(50,120),(200,120),colors['green'],3,cv2.LINE_AA,0,0.3)
cv2.arrowedLine(image,(50,200),(200,200),colors['blue'],3,8,0,0.3)
plt.imshow(image)
plt.title('Dibujando flechas')
plt.show()


# In[9]:


image=np.zeros((300,300,3),dtype='uint8')
image[:]=colors['light_gray']
cv2.ellipse(image,(80,80),(60,40),0,0,360,colors['red'],-1)
cv2.ellipse(image,(80,200),(80,40),0,0,360,colors['green'],3)
cv2.ellipse(image,(200,200),(10,40),0,0,180,colors['yellow'],3)
cv2.ellipse(image,(200,100),(10,40),0,0,270,colors['cyan'],3)
cv2.ellipse(image,(250,100),(20,40),45,0,360,colors['gray'],3)
cv2.ellipse(image,(250,250),(30,30),0,0,360,colors['magenta'],3)
plt.imshow(image)
plt.title('Dibujando elipces')
plt.show()


# In[10]:


image=np.zeros((300,300,3),dtype='uint8')
image[:]=colors['light_gray']
pts_triangulo=np.array([[250,5],[220,80],[280,80]],np.int32)
pts_pentagono=np.array([[100,100],[20,170],[50,250],[150,250],[180,170]],np.int32)
pts_pentagono=np.reshape(pts_pentagono,[-1,1,2])
pts_triangulo=np.reshape(pts_triangulo,[-1,1,2])
cv2.polylines(image,[pts_triangulo],True,colors['green'],3)
cv2.polylines(image,[pts_pentagono],True,colors['red'],3)
plt.imshow(image)
plt.title('Dibujando poligonos')
plt.show()


# In[11]:


image=np.zeros((300,300,3),dtype='uint8')
image[:]=colors['light_gray']

shift=2
factor = 2**shift

cv2.circle(image,(int(round(299.9*factor)),int(round(299.9*factor))),300*factor,colors
['red'],1,8,shift=2)
cv2.circle(image,(int(round(299.9*factor)),int(round(299.9*factor))),300*factor,colors['blue'],1,8,shift=3)
plt.imshow(image)
plt.title('Dibujando circulos con float')
plt.show()


# In[12]:


# Escribiendo texto
image=np.zeros((130,490,3),dtype='uint8')
#image[:]=colors['white']
image.fill(255) # Para ajustar directamente a un color 

cv2.putText(image,'Mastering OpenCV4 with Python',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,colors['red'],3,cv2.LINE_4)
cv2.putText(image,'Mastering OpenCV4 with Python',(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.9,colors['blue'],2,cv2.LINE_8,)
cv2.putText(image,'Mastering OpenCV4 with Python',(10,110),cv2.FONT_HERSHEY_SIMPLEX,0.9,colors['magenta'],1,cv2.LINE_AA)
plt.imshow(image)
plt.title('Dibujando texto')
plt.show()


# In[13]:


"""FONT_HERSHEY_SIMPLEX = 0
   FONT_HERSHEY_PLAIN = 1
   FONT_HERSHEY_DUPLEX = 2
   FONT_HERSHEY_COMPLEX = 3
   FONT_HERSHEY_TRIPLEX = 4
   FONT_HERSHEY_COMPLEX_SMALL = 5
   FONT_HERSHEY_SCRIPT_SIMPLEX = 6
   FONT_HERSHEY_SCRIPT_COMPLEX = 7 """
image=np.zeros((130,490,3),dtype='uint8')
image[:]=colors['light_gray']
font=cv2.FONT_HERSHEY_SIMPLEX
font_scale=2.5
thickness=5
text='DataScience'
radius=10
ret,baseline=cv2.getTextSize(text,font,font_scale,thickness=thickness)
text_w,text_h=ret # Se obtiene el ancho y alto del texto
text_x=int(round((image.shape[1] - text_w)/2))
text_y=int(round((image.shape[0] + text_h)/2))
cv2.circle(image,(text_x,text_y),radius,colors['blue'],-1)
cv2.rectangle(image,(text_x,text_y+baseline),(text_x+text_w-thickness,text_y-text_h),colors['red'],thickness=thickness)
cv2.circle(image,(text_x,text_y + baseline),radius,colors['green'],-1)
cv2.line(image,(text_x,text_y+int(round(thickness/2))),(text_x+text_w-thickness,text_y+int(round(thickness/2))),colors['yellow'],thickness)
cv2.putText(image,text,(text_x,text_y),font,font_scale,colors['cyan'],thickness)
plt.imshow(image)
plt.title('Dibujando circulos y rectangulos con texto')
plt.show()


# In[ ]:


# Dibujos dinamicos por eventos del mouse
image=np.zeros((300,300,3),dtype='uint8')
image[:]=colors['white']

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('Event: EVENT_LBUTTONDBLCLK')
        cv2.circle(image,(x,y),10,colors['red'],-1)
        cv2.imshow('Circle',image)
    elif event == cv2.EVENT_MOUSEMOVE:
        print('Event:EVENT_MOUSEMOVE')
    elif event == cv2.EVENT_LBUTTONUP:
        print('Event: EVENT_LBUTTONUP')
    elif event == cv2.EVENT_LBUTTONDOWN:
        print('Event: EVENT_LBUTTONDOWN')

cv2.namedWindow('Image mouse')
cv2.setMouseCallback('Image mouse',draw_circle)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


# Dibujando texto en eventos dinamicos del mouse
image=np.zeros((520,500,3),dtype='uint8')
image[:]=colors['black']

text='Double left click : add a circle'
text_1='Simple right click : delete last circle'
text_2='Double right click : delete all circle'
text_3='Press (q) to exit'

font=cv2.FONT_HERSHEY_PLAIN
font_scale=1.5
cv2.putText(image,text,(10,350),font,font_scale,colors['white'],thickness=1)
cv2.putText(image,text_1,(10,400),font,font_scale,colors['white'],thickness=1)
cv2.putText(image,text_2,(10,450),font,font_scale,colors['white'],thickness=1)
cv2.putText(image,text_3,(10,500),font,font_scale,colors['white'],thickness=1)
pos_circulos=[]
def draw_circles(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image,(x,y),10,colors['blue'],-1)
        pos_circulos.append((x,y))
        cv2.imshow('Circle',image)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Borrar el ultimo  circulo
        valor=pos_circulos.pop()
        cv2.circle(image,valor,10,colors['black'],-1)
        cv2.imshow('Circle',image)

    elif event == cv2.EVENT_RBUTTONDBLCLK:
        # Borrar todos los circulos
        for i in pos_circulos:
            cv2.circle(image,i,10,colors['black'],-1)
        cv2.imshow('Circle',image)
        pos_circulos.clear()

cv2.namedWindow('Image mouse')
cv2.setMouseCallback('Image mouse',draw_circles)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


# Capturando eventos co Matplotlib
image=np.zeros((400,400,3),dtype='uint8')
image[:]=colors['light_gray']

def update_image():
    image_RGB=image[:,:,::-1]
    plt.imshow(image_RGB)
    fig.canvas.draw()
def click_mouse_event(event):
    cv2.circle(image,(int(round(event.xdata)),int(round(event.ydata))),30,colors['blue'],cv2.FILLED)
    update_image()
    
fig=plt.figure()
fig.add_subplot(111)
update_image()
fig.canvas.mpl_connect('button_press_event',click_mouse_event)
plt.show()


# In[14]:


# Dibujos avanzados
import datetime,math

image=np.zeros((700,700,3),dtype='uint8')
image[:]=colors['light_gray']
hours_orig = np.array(
    [(620, 320), (580, 470), (470, 580), (320, 620), (170, 580), (60,
470), (20, 320), (60, 170), (169, 61), (319, 20),
     (469, 60), (579, 169)]
)
hours_dest = np.array(
    [(600, 320), (563, 460), (460, 562), (320, 600), (180, 563), (78,
460), (40, 320), (77, 180), (179, 78), (319, 40),
     (459, 77), (562, 179)]
)
def array_to_tuple(tupla):
    valor=tuple(tupla)
    return valor

for i in range(0,12):
    cv2.line(image,array_to_tuple(hours_orig[i]),array_to_tuple(hours_dest[i]),colors['black'],3)


cv2.rectangle(image,(150,175),(490,270),colors['light_gray'],-1)
cv2.putText(image,'Samsung',(220,200),1,3,colors['blue'],3,cv2.LINE_AA)

date_time_now=datetime.datetime.now()
time_now=date_time_now.time()
hour=math.fmod(time_now.hour,12)
minute=time_now.minute
second=time_now.second

second_ang=math.fmod(second * 6 + 270,360)
minute_ang=math.fmod(minute * 6 + 270,360)
hour_ang=math.fmod((hour * 30)+(minute/2)+270,360)

second_x=round(320 + 310 * math.cos(second_ang*3.14/180))
second_y=round(320 + 310 * math.sin(second_ang*3.14/180))
cv2.line(image,(320,320),(second_x,second_y),colors['blue'],2)

minute_x=round(320 + 260 * math.cos(minute_ang*3.14/180))
minute_y=round(320 + 260 * math.sin(minute_ang*3.14/180))
cv2.line(image,(320,320),(minute_x,minute_y),colors['blue'],8)

hour_x=round(320 + 220 * math.cos(hour_ang*3.14/180))
hour_y=round(320 + 220 * math.sin(hour_ang*3.14/180))
cv2.line(image,(320,320),(hour_x,hour_y),colors['blue'],10)

cv2.circle(image,(320,320),310,colors['dark_gray'],8)
cv2.imshow('Reloj Analogico',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




