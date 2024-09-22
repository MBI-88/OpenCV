# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:28:24 2021

@author: MBI
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = (12,10)
np.random.seed(1)
#%%
"""
Momentun de una imagen puede ser visto como el pesado promedio de la intensidad de los pixesls
de una imagen, codificando algunas propiedades interesantes de una imagen.En esta medida momentun
de una imagen es de uso para describir algunas propiedades de los contornos detectados ejemplo el 
centro de masa del objeto , el area del objeto entre otros.
"""
def build_sample_image2():
    img = np.ones(shape=(500,500,3),dtype='uint8') * 70
    cv2.rectangle(img,(100,100),(300,300),(255,0,255),-1)
    cv2.rectangle(img,(150,150),(250,250),(70,70,70),-1)
    cv2.circle(img,(400,400),100,(255,255,0),-1)
    cv2.circle(img,(400,400),50,(70,70,70),-1)
    return img


image = build_sample_image2()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
M = cv2.moments(contours[0])
print(M)

"""
Hay 3 tipos diferentes de momentum (mji,muji,nuji).

El momentum espacial mji es computado como sigue:
    mji = Sumatoria_x,y (array(x,y) * x^j * y^i)

El momentum central muji  es computado como sigue:
    muji = Sumatoria_x,y (array(x,y) * (x - X)^j * (y - Y)^i))
    donde X,Y son:
        X = m10/m00 , Y = m01/m00 . Esta ecuacion corresponde al centro de masa (centroides).
        
Momentum central normalizado nuji es compudato como sigue:
    nuji = muji/m00^(i + j)/2 + 1

El valor para el siguiente momentum es calculado como sigue:
    mu00 = m00 , nu00 = 1 , nu10 = mu10 = mu01 = mu10 = 0.
    Estos momentum no son almacenados.
"""
#%%
# Variables de objetos basados en momentum

print('Contour area: {}'.format(cv2.contourArea(contours[0])))
print('Contour area: {}'.format(M['m00']))
print('\n')
print('Center X: {}'.format(round(M['m10']/M['m00'])))
print('Center Y: {}'.format(round(M['m01']/M['m00'])))
#%%
"""
Roundness K es la medida de que tan cercano es el acercamiento de un contorno al contorno del 
circulo perfecto. El roundnes de un contorno pude ser calculado  acorde a la siguiente formula

K = P^2/(A * 4 * pi)

P: Es el perimetro de el contorno 
A: Es el correspondiente area 

En caso de un circulo perfecto el resultado sera 1; el mas alto valor obtenido , menos circular sera

"""
def roundness(contour,moments):
    "Calculates the roundness of a contour"
    
    length = cv2.arcLength(contour,True) # Perimetro
    K = (length * length)/(moments['m00'] * 4 * np.pi)
    return K

"""
Elongacion es la medida de que tan elongado puede ser el contorno. La elongacion (e) puede ser directamente
derivada de el semi-major y semi-minor ejes a y b del objeto, acorde a la siguiente formula:
    
    e = Raiz^2((a^2 - b^2)/b^2)
    
"""
def eccentricity_from_ellipse(contour):
    "Calculates the eccentricity fitting an ellipse from a contours"
    
    (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
    a = ma/2
    b = MA/2
    ecc = np.sqrt(a**2 - b**2)/a
    return ecc

# Calculando la elongacion a partir del momentum

def eccentricity_from_moments(moments):
    "Calculates the eccentricity from themoments of the contour"
    
    a1 = (moments['m20'] + moments['mu02'])/2
    a2 = np.sqrt(4 * moments['nu11']**2 + (moments['mu20'] - moments['mu02'])**2)/2
    ecc = np.sqrt(1 - (a1 - a2)/(a1 + a2))
    return ecc
"""
El radio de aspecto puede ser calaculado  facilmente basado sobre las dimensiones de las 
bondaridades minimas del rectangulo. El radio  de aspecto es el radio de ancho mas grande de 
las bondaridades del rectangulo del contorno.
"""
def aspect_ratio(contour):
    "Returns the aspect ratio of the contour based on the dimensions of the bounding rect"
    
    x,y,w,h = cv2.boundingRect(contour)
    res = float(w)/h
    return res

#%%
def get_one_contour():
    "Returns a fixed cotour"
    cnts = [np.array(
            [[[600, 320]], [[563, 460]], [[460, 562]], [[320, 600]], [[180, 563]],
            [[78, 460]], [[40, 320]], [[77, 180]], [[179, 78]], [[319, 40]], [[459,
            77]], [[562, 179]]], dtype=np.int32)]
    return cnts

def draw_contour_points(img,cnts,color,centroi):
    for cnt in cnts:
        squeeze = np.squeeze(cnt)
        for p in squeeze:
            p = array_to_tuple(p)
            cv2.circle(img,p,10,color,-1)
    
    cv2.circle(img,centroi,10,(255,0,0),-1)
    return img

def array_to_tuple(arr):
    return tuple(arr.reshape(1,-1)[0])

def draw_outline_contour(img,cnts,color):
    for cn in cnts:
        cv2.drawContours(img,[cn],0,color)
    return img


def draw_area_contour(img,centroi,area):
    ratio = np.sqrt((area/np.pi))
    ratio = int(ratio)
    cv2.circle(img,centroi,ratio,(0,255,0),-1)
    return img

def get_centroi(moments):
    X = round(moments['m10']/moments['m00'])
    Y = round(moments['m01']/moments['m00'])
    return X,Y
    
#%%
img = np.zeros(shape=(640,640,3),dtype='uint8')
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()
cont = get_one_contour()
M = cv2.moments(cont[0])
centroi = get_centroi(M)
ratio = aspect_ratio(cont[0])
elon = eccentricity_from_ellipse(cont[0])
rou = roundness(cont[0],M) 

img_centroi = draw_contour_points(img1,cont,(255,120,0),centroi)
img_area = draw_area_contour(img2,centroi,M['m00'])
img_round_ecc = draw_outline_contour(img3,cont,(255,0,125))

plt.subplot(131)
plt.imshow(img_centroi)
plt.title('Centroi {}'.format(centroi))
plt.axis('off')
plt.subplot(132)
plt.imshow(img_area)
plt.title('Size: {} Aspect_ratio: {}'.format(M['m00'],ratio))
plt.axis('off')
plt.subplot(133)
plt.imshow(img_round_ecc)
plt.title('Roundness: {} Eccentricity: {}'.format(rou,elon))
plt.axis('off')
plt.show()
#%%

def roundness(contour, moments):
    """Calculates the roundness of a contour"""

    length = cv2.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * np.pi)
    return k


def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """Gives the coordinates to draw centered"""

    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)


def eccentricity_from_ellipse(contour):
    """Calculates the eccentricity fitting an ellipse from a contour"""

    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)

    a = ma / 2
    b = MA / 2

    ecc = np.sqrt(a ** 2 - b ** 2) / a
    return ecc


def eccentricity_from_moments(moments):
    """Calculates the eccentricity from the moments of the contour"""

    a1 = (moments['mu20'] + moments['mu02']) / 2
    a2 = np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2) / 2
    ecc = np.sqrt(1 - (a1 - a2) / (a1 + a2))
    return ecc


def build_image_ellipses():
    """Draws ellipses in the image"""

    img = np.zeros((500, 600, 3), dtype="uint8")
    cv2.ellipse(img, (120, 60), (100, 50), 0, 0, 360, (255, 255, 0), -1)
    cv2.ellipse(img, (300, 60), (50, 50), 0, 0, 360, (0, 0, 255), -1)
    cv2.ellipse(img, (425, 200), (50, 150), 0, 0, 360, (255, 0, 0), -1)
    cv2.ellipse(img, (550, 250), (20, 240), 0, 0, 360, (255, 0, 255), -1)
    cv2.ellipse(img, (200, 200), (150, 50), 0, 0, 360, (0, 255, 0), -1)
    cv2.ellipse(img, (250, 400), (200, 50), 0, 0, 360, (0, 255, 255), -1)
    return img


def draw_contour_outline(img, cnts, color, thickness=1):
    """Draws contours outlines of each contour"""

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(14, 6))
plt.suptitle("Eccentricity", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
# image = build_sample_image_2()
image = build_image_ellipses()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get a binary image:
ret, thresh = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)

# Find contours using the thresholded image:
# Note: cv2.findContours() has been changed to return only the contours and the hierarchy
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Show the number of detected contours:
print("detected contours: '{}' ".format(len(contours)))

# Create a copy to show the results:
img_numbers = image.copy()

for contour in contours:
    # Draw the contour:
    draw_contour_outline(image, [contour], (255, 255, 255), 5)

    # Compute the moments of the contour:
    M = cv2.moments(contour)

    # Calculate the roundness:
    k = roundness(contour, M)
    print("roundness: '{}'".format(k))

    # Calculate eccentricy using the two provided formulas:
    em = eccentricity_from_moments(M)
    print("eccentricity: '{}'".format(em))
    ee = eccentricity_from_ellipse(contour)
    print("eccentricity: '{}'".format(ee))

    # Get centroid of the contour:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    # Ge get the text to draw:
    text_to_draw = str(round(em, 3))

    # Get the position to draw:
    (x, y) = get_position_to_draw(text_to_draw, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, 3)

    # Write the name of shape on the center of shapes:
    cv2.putText(img_numbers, text_to_draw, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

# Plot the images:
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(img_numbers, "ellipses eccentricity", 2)

# Show the Figure:
plt.show()
#%%
# Momentum invariante  Hu
"""
El momento invariante Hu es invariante  con respecto  a la traslacion,scala y rotacion en todos 
los momentos excepto en el septimo con respecto a la reflexion. OpenCv provee cv2.HuMoments() para calcular el septimo momentun
invariante Hu.

Definicion del metodo:
    cv2.HuMoments(m,[,hu])-> hu
    m: corresponde  al momentum calculado con cv2.momnets()
    hu: corresponde a el septimo momentum invariante Hu

Definicion de Hu moment invariants:
    hu[0] = n20 + n02
    hu[1] = (n20 - n02)^2 + 4n11^2
    hu[2] = (n30 - 3n12)^2 + (3n21 - n03)^2
    hu[3] = (n30 + n12)^2 + (n21 + n03)^2
    ......
    
    
"""
image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/shape_features.png')
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(gray_image,70,255,cv2.THRESH_BINARY)
M = cv2.moments(thresh,True)
print('Moments: {}\n'.format(M))


x,y = get_centroi(M)

HuM = cv2.HuMoments(M)
print('Hu moments: {}\n'.format(HuM))

contour,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
M2 = cv2.moments(contour[0])
print('Moments: {}\n'.format(M2))

x2,y2 = get_centroi(M2)

HuM2 = cv2.HuMoments(M2)
print('Hu momnents: {}\n'.format(HuM2))

print('Centroids X = {}, Y = {}\n'.format(x,y))
print('Centroids X2 = {}, Y2 = {}\n'.format(x2,y2))

for cont in contour:
    draw_contour_outline(image, [cont], (255,100,0),10)

cv2.circle(image,(x,y),20,(255,0,74),-1)
cv2.circle(image,(x2,y2),20,(74,0,255),-1)

fig = plt.figure(figsize=(10,4))
plt.suptitle('Hu moments',fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')
plt.imshow(image)
plt.title('Detected contour and centroid')
plt.axis('off')
plt.show()


#%%
def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

image_1 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/shape_features.png')
image_2 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/shape_features_rotation.png')
image_3 = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/shape_features_reflection.png')

gray_1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
gray_3 = cv2.cvtColor(image_3,cv2.COLOR_BGR2GRAY)

ret1,thresh1 = cv2.threshold(gray_1,70,255,cv2.THRESH_BINARY)
ret2,thresh2 = cv2.threshold(gray_2,70,255,cv2.THRESH_BINARY)
ret3,thresh3 = cv2.threshold(gray_3,70,255,cv2.THRESH_BINARY)

HuM1 = cv2.HuMoments(cv2.moments(thresh1,True)).flatten()
HuM2 = cv2.HuMoments(cv2.moments(thresh2,True)).flatten()
HuM3 = cv2.HuMoments(cv2.moments(thresh3,True)).flatten()

print('Hu moments (original): {}'.format(HuM1))
print('Hu moments (rotation): {}'.format(HuM2))
print('Hu moments (reflection): {}'.format(HuM3))

fig = plt.figure(figsize=(10,4))
plt.suptitle('Hu moments invariants properties',fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

show_img_with_matplotlib(image_1,'Original',1)
show_img_with_matplotlib(image_2,'Rotation',2)
show_img_with_matplotlib(image_3,'Reflection',3)

plt.show()



#%%