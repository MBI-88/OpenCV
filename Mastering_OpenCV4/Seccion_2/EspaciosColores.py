#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,15)


# ***Espacios de colores.***
# ***Mostrando espacios de colores***
# ***El espacio de color RGB es un espacio aditivo donde un color especifico es representado por el rojo,verde y azul.La vision humana trabaja de forma similar, este espacio de colores es la forma apropiada para mostrar graficas.***
# ***El espacio CIELAB representa un color especifico con tres valores numericos L-sin brillo,A-componente verde-rojo,B representa el componente azul-amarillo.Este espacio de colores es usado en algunos algoritmos de procesamiento de imagenes***
# ***HLS/HSV son dos espacios de colores donde solo un canal (H) es usado para describir el color, haciendolo muy intuitivo para especificar colores.En estos modelos de colores, la separacion de los componentes de luminancia tienen ventaja cuando se aplican en tecnicas de procesado.***
# ***YCbCr es una familia de espacios de colores usado en video y sistemas photograficos digitales, representan colores en terminos de componentes chroma (Y) y dos componentes chrominace/chroma (Cb y Cr).Este espacio de colores es muy popular en la segmentaacion de imagenes, basado en modelos de colores derivados de YCbCr***

# In[2]:


image_original=cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/cat_dog_images/cat-02.jpg')

image_RGB=cv2.cvtColor(image_original,cv2.COLOR_BGR2RGB)
image_CIELAB=cv2.cvtColor(image_original,cv2.COLOR_BGR2LAB)
image_HSL=cv2.cvtColor(image_original,cv2.COLOR_BGR2HLS)
image_HSV=cv2.cvtColor(image_original,cv2.COLOR_BGR2HSV)
image_YCbCr=cv2.cvtColor(image_original,cv2.COLOR_BGR2YCrCb)

plt.subplot(321)
plt.imshow(image_original)
plt.title('Image BGR')
plt.axis('off')
plt.subplot(322)
plt.imshow(image_RGB)
plt.title('Image RGB')
plt.axis('off')
plt.subplot(323)
plt.imshow(image_CIELAB)
plt.title('Image CIELAB')
plt.axis('off')
plt.subplot(324)
plt.imshow(image_HSL)
plt.title('Image HSL')
plt.axis('off')
plt.subplot(325)
plt.imshow(image_HSV)
plt.title('Image HSV')
plt.axis('off')
plt.subplot(326)
plt.imshow(image_YCbCr)
plt.title('Image YCbCr')
plt.axis('off')
plt.show()


# In[15]:


# Segmentacion de piel en diferentes espacios de colores
# Segun papper de desarrolladores

image_bgr=cv2.imread('lenna.png')


#  Segmentacion en HSV
low_hsv=np.array([0,40,50],dtype='uint8')
upper_hsv=np.array([40,200,250],dtype='uint8')
# Valores originales del papper son [0,48,80] y [20,255,255]
image_hsv=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
skin_region=cv2.inRange(image_hsv,low_hsv,upper_hsv)

# Segmentacion en HSV 2
low_hsv_2=np.array([10,70,148],dtype='uint8')
upper_hsv_2=np.array([135,130,200],dtype='uint8')
# Valores originales del papper son [0,50,0] y [120,150,255]
image_hsv_2=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
skin_region_2=cv2.inRange(image_bgr,low_hsv_2,upper_hsv_2)

# Segmentacion en YCbCr
low_ycbcr=np.array([10,20,100],dtype='uint8') 
# Valores del papper [0,133,77] y [255,173,127]
upper_ycbcr=np.array([120,80,150],dtype='uint8')

image_ycbcr=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2YCR_CB)
skin_ycbcr=cv2.inRange(image_bgr,low_ycbcr,upper_ycbcr)

# Segmentacion en BGR
def bgr_skin(b,g,r):
    "Reglas basado en el papper de segmentacion  RGB-H-CbCr para color de piel  en deteccion de piel humana"

    e1=bool((r>95) and (g>40) and (b>20) and ((max(r,max(g,b))-min(r,min(g,b)))>15) and (abs(int(r)-int(g))>15) and (r>g) and (r>b) )
    e2=bool((r>200) and (g>210) and (b>170) and (abs(int(r)-int(g))<=15) and (r>b) and (g>b))
    return e1 or e2

h=image_bgr.shape[0]
w=image_bgr.shape[1]
res=np.zeros((h,w),dtype='uint8')
# Solamente los pixeles de piel seran establecidos a blanco (255)
for y in range(0,h):
    for x in  range(0,w):
        (b,g,r)=image_bgr[y,x]
        if bgr_skin(b,g,r):
            res[y,x]=255

image_rgb=image_bgr[:,:,::-1]

plt.subplot(321)
plt.imshow(image_rgb)
plt.title('RGB')
plt.axis('off')
plt.subplot(322)
plt.imshow(skin_region)
plt.title('HSV')
plt.axis('off')
plt.subplot(323)
plt.imshow(skin_region_2)
plt.title('HSV_2')
plt.axis('off')
plt.subplot(324)
plt.imshow(skin_ycbcr)
plt.title('YCbCr')
plt.axis('off')
plt.subplot(325)
plt.imshow(res)
plt.title('BGR')
plt.axis('off')
plt.show()


# In[5]:


# Mapas de Colores
"""COLORMAP_AUTUMN = 0
   COLORMAP_BONE = 1
   COLORMAP_JET = 2
   COLORMAP_WINTER = 3
   COLORMAP_RAINBOW = 4
   COLORMAP_OCEAN = 5
   COLORMAP_SUMMER = 6
   COLORMAP_SPRING = 7
   COLORMAP_COOL = 8
   COLORMAP_HSV = 9
   COLORMAP_HOT = 11
   COLORMAP_PINK = 10
   COLORMAP_PARULA = 12"""

image_gray=cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)

image_color_hsv=cv2.applyColorMap(image_gray,cv2.COLORMAP_HSV)
imge_color_autum=cv2.applyColorMap(image_gray,0)
image_color_jet=cv2.applyColorMap(image_gray,2)
image_color_cool=cv2.applyColorMap(image_gray,8)

plt.subplot(221)
plt.imshow(image_color_hsv)
plt.title('ImageColor HSV')
plt.axis('off')
plt.subplot(222)
plt.imshow(imge_color_autum)
plt.title('ImageColor autum')
plt.axis('off')
plt.subplot(223)
plt.imshow(image_color_jet)
plt.title('ImageColor jet')
plt.axis('off')
plt.subplot(224)
plt.imshow(image_color_cool)
plt.title('ImageColor cool')
plt.axis('off')
plt.show()


# In[6]:


# Mapas de colores personalizados
image_=cv2.imread('shades.png',cv2.IMREAD_GRAYSCALE)


lut=np.zeros((256,1,3),dtype='uint8')
for i in range(255):# Para aplicarlo a cv2.applayColor()
    valor=np.random.random_integers(10,255)
    lut[:,0,0]=[valor]
    valor_1=np.random.random_integers(0,200)
    lut[:,0,1]=[valor_1]
    valor_2=np.random.random_integers(95,195)
    lut[:,0,2]=[valor_2]
    
image_color=cv2.applyColorMap(image_,lut)

lut=np.zeros((256,3),dtype='uint8')
for i in range(0,255):# Para aplicar a cv2.lut
    valor=np.random.random_integers(10,255)
    lut[:,0]=[valor]
    valor_1=np.random.random_integers(0,200)
    lut[:,1]=[valor_1]
    valor_2=np.random.random_integers(95,195)
    lut[:,2]=[valor_2]

s0,s1=image_.shape[:2]
img_color=np.empty(shape=(s0,s1,3),dtype='uint8')
for i in range(3):
    img_color[...,i]=cv2.LUT(image_,lut[:,i])

plt.subplot(131)
plt.imshow(image_,cmap='Greys')
plt.title('Original')
plt.axis('off')
plt.subplot(132)
plt.imshow(image_color)
plt.title('cv2.applyColorMap()')
plt.axis('off')
plt.subplot(133)
plt.imshow(img_color)
plt.title('cv2.LUT()')
plt.axis('off')
plt.show()
# Nota: Este ejemplo  es solo para mostrar el uso de la funcion lut y applaColor con mapas de colores personalizados.


# In[7]:


# Variante optima para personalizar un mapa de colores

dict_color={0:'blue',1:'green',2:'red'}

def build_lut(cmap):
    lut=np.empty(shape=(256,3),dtype='uint8')
    print('----------')
    print(cmap)
    print('----------')
    max=256
    lastval,lastcol=cmap[0]
    for step,col in cmap[1:]:
        val=int(step*max)
        for  i in range(3):
            print("{}  : np.linspace('{}','{}','{}' - '{}' = '{}')".format(dict_color[i],lastcol[i],col[i],val,lastval,val-lastval))
            lut[lastval:val,i]=np.linspace(lastcol[i],col[i],val-lastval)
        lastcol=col
        lastval=val
    return lut

def apply_color_map_1(gray,cmap):
    lut=build_lut(cmap)
    s0,s1=gray.shape
    out=np.empty(shape=(s0,s1,3),dtype='uint8')
    for i in range(3):
        out[...,i]=cv2.LUT(gray,lut[:,i])
    return out

def apply_color_map_2(gray,cmap):
    lut=build_lut(cmap)
    lut_reshape=np.reshape(lut,(256,1,3))
    im_color=cv2.applyColorMap(gray,lut_reshape)
    return im_color

def show_matplotlib(color_image,tittle,pos):
    img_RGB=color_image[:,:,::-1]
    ax=plt.subplot(2,3,pos)
    plt.imshow(img_RGB)
    plt.title(tittle)
    plt.axis('off')

gray_img=cv2.imread('shades.png',cv2.IMREAD_GRAYSCALE)
plt.suptitle('Custom color maps based on key color',fontsize=14,fontweight='bold')
show_matplotlib(cv2.cvtColor(gray_img,cv2.COLOR_GRAY2BGR),'gray',1)
custom_1=apply_color_map_1(gray_img,((0,(255,0,255)),(0.25,(255,0,180)),(0.5,(255,0,120)),(0.75,(255,0,60)),(1.0,(255,0,0))))
custom_2=apply_color_map_1(gray_img,((0,(0,255,128)),(0.25,(128,184,64)),(0.5,(255,128,0)),(0.75,(64,128,224)),(1.0,(0,128,255))))
custom_3=apply_color_map_2(gray_img,((0,(255,0,255)),(0.25,(255,0,180)),(0.5,(255,0,120)),(0.75,(255,0,60)),(1.0,(255,0,0))))
custom_4=apply_color_map_2(gray_img,((0,(0,255,128)),(0.25,(128,184,64)),(0.5,(255,128,0)),(0.75,(64,128,224)),(1.0,(0,128,255))))

show_matplotlib(custom_1,'custom_1 using cv2.LUT()',2)
show_matplotlib(custom_2,'custom_2 using cv2.LUT()',3)
show_matplotlib(custom_3,'custom_3 using cv2.applyColorMap()',5)
show_matplotlib(custom_4,'custom_4 using cv2.applyColorMap()',6)
plt.show()
# Nota: Para obtener los colores priarios usando np.linspace


# In[8]:


# Mostrando legenda del mapa de colores

def build_lut(cmap):
    lut=np.empty(shape=(256,3),dtype='uint8')
    max=256
    lastval,lastcol=cmap[0]
    for step,col in cmap[1:]:
        val=int(step*max)
        for  i in range(3):
            lut[lastval:val,i]=np.linspace(lastcol[i],col[i],val-lastval)
        lastcol=col
        lastval=val
    return lut

def apply_color_map_1(gray,cmap):
    lut=build_lut(cmap)
    s0,s1=gray.shape
    out=np.empty(shape=(s0,s1,3),dtype='uint8')
    for i in range(3):
        out[...,i]=cv2.LUT(gray,lut[:,i])
    return out

def apply_color_map_2(gray,cmap):
    lut=build_lut(cmap)
    lut_reshape=np.reshape(lut,(256,1,3))
    im_color=cv2.applyColorMap(gray,lut_reshape)
    return im_color

def show_matplotlib(color_image,tittle,pos):
    img_RGB=color_image[:,:,::-1]
    ax=plt.subplot(2,2,pos)
    plt.imshow(img_RGB)
    plt.title(tittle)
    plt.axis('off')

def build_lut_image(cmap,height):
    lut=build_lut(cmap)
    image=np.repeat(lut[np.newaxis,...],height,axis=0)
    return image

image_lenna=cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)

custom_1=apply_color_map_1(image_lenna,((0,(255,0,255)),(0.25,(255,0,180)),(0.5,(255,0,120)),(0.75,(255,0,60)),(1.0,(255,0,0))))
custom_2=apply_color_map_1(image_lenna,((0,(0,255,128)),(0.25,(128,184,64)),(0.5,(255,128,0)),(0.75,(64,128,224)),(1.0,(0,128,255))))

lengend_1=build_lut_image(((0,(255,0,255)),(0.25,(255,0,180)),(0.5,(255,0,120)),(0.75,(255,0,60)),(1.0,(255,0,0))),20)
lengend_2=build_lut_image(((0,(0,255,128)),(0.25,(128,184,64)),(0.5,(255,128,0)),(0.75,(64,128,224)),(1.0,(0,128,255))),20)

plt.suptitle('Custom color maps based on key color',fontsize=14,fontweight='bold')
show_matplotlib(lengend_1,"",1)
show_matplotlib(custom_1,"",3)
show_matplotlib(lengend_2,"",2)
show_matplotlib(custom_2,"",4)
plt.show()


# In[ ]:




