# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:23:30 2021

@author: MBI
"""
import cv2,dlib,cvlib,face_recognition
import numpy as np
import matplotlib.pyplot as plt
#%%
# Detectando rasgos  faciales con opencv
"""
Opencv ofrece 3 diferentes implementaciones de deteccion de rasgos faciales basados en los paipers:
    . FacemarkLBF
    . FacemarkKamezi
    . FacemarkAAM

Estas implementaciones fallan por problemas en la cofiguracion del codigo en las librerias de C++


image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/face_test.png',0)

cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
faces = cas.detectMultiScale(image,1.5,5)
print('Faces: {}\n'.format(faces))


print("Testing LBF\n")
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")
ok,landmarks = facemark.fit(image,faces)
print("Landmark: {} {}\n".format(ok,landmarks))

print("Testing AAM\n")
facemark = cv2.face.createFacemarkAAM()
facemark.loadModel("aam.xml")
ok,landmarks = facemark.fit(image,faces)
print("Landmarks AAM: {} {}\n".format(ok,landmarks))

print("Testing Kazemi\n")
facemark = cv2.face.createFacemarkKazemi()
facemark.loadModel("face_landmark_model.dat")
ok,landmarks = facemark.fit(image,faces)
print("Landmarks Kazemi: {} {}".format(ok,landmarks))


Para la solucion se debe  hacer lo siguiente: Modificar el codigo fuente de C++ del metodo fit()

Este es el codigo 

// C++ code
bool FacemarkLBFImpl::fit( InputArray image, InputArray roi,
OutputArrayOfArrays _landmarks )
{
    // FIXIT
    std::vector<Rect> & faces = *(std::vector<Rect> *)roi.getObj();
    if (faces.empty()) return false;
    std::vector<std::vector<Point2f> > & landmarks =
        *(std::vector<std::vector<Point2f> >*) _landmarks.getObj();
    landmarks.resize(faces.size());
    for(unsigned i=0; i<faces.size();i++){
        params.detectROI = faces[i];
        fitImpl(image.getMat(), landmarks[i]);
    }
    return true;
}

Esta es la modificacion:

// C++ code
bool FacemarkLBFImpl::fit( InputArray image, InputArray roi,
OutputArrayOfArrays _landmarks )
{
    Mat roimat = roi.getMat();
    std::vector<Rect> faces = roimat.reshape(4,roimat.rows);
    if (faces.empty()) return false;
    std::vector<std::vector<Point2f> > landmarks(faces.size());
    for (unsigned i=0; i<faces.size();i++){
        params.detectROI = faces[i];
            fitImpl(image.getMat(), landmarks[i]);
    }
    if (_landmarks.isMatVector()) { // python
        std::vector<Mat> &v = *(std::vector<Mat>*) _landmarks.getObj();
        for (size_t i=0; i<faces.size(); i++)
            v.push_back(Mat(landmarks[i]));
    } else { // c++, java
        std::vector<std::vector<Point2f> > &v = *
(std::vector<std::vector<Point2f> >*) _landmarks.getObj();
        v = landmarks;
    }
    return true;
}
"""
#%%
# Detectando rasgos faciales con dlib

Jawline_points = list(range(0,17))
Right_Eyebrown_points = list(range(17,22))
Left_Eyebrown_points = list(range(22,27))
Nose_Bridge_points = list(range(27,31))
Lower_nose_points = list(range(31,36))
Right_Eye_points = list(range(36,42))
Left_Eye_points = list(range(42,48))
Mouth_Outline_points = list(range(48,61))
Mouth_Inner_points = list(range(61,68))
All_points = list(range(0,68))

def draw_shape_lines_range(np_shape,image,range_points,is_closed=False):
    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display,dtype = "int32")
    cv2.polylines(image,[points],is_closed,(255,255,0),thickness=1,lineType=cv2.LINE_8)

def draw_shape_points_pos_range(np_shape,image,points):
    np_shape_display = np_shape[points]
    draw_shape_points_pos(np_shape_display,image)

def draw_shape_points_pos(np_shape,image):
    for idx,(x,y) in enumerate(np_shape):
        cv2.putText(image,str(idx+1),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255))
        cv2.circle(image,(x,y),2,(0,255,0),-1)

def draw_shape_points_range(np_shape,image,points):
    np_shape_display = np_shape[points]
    draw_shape_points(np_shape_display,image)

def draw_shape_points(np_shape,image):
    for (x,y) in np_shape:
        cv2.circle(image,(x,y),2,(0,255,0),-1)

def draw_shape_line_all(np_shape,image):
    
    draw_shape_lines_range(np_shape,image,Jawline_points)
    draw_shape_lines_range(np_shape,image,Right_Eyebrown_points)
    draw_shape_lines_range(np_shape,image,Left_Eye_points)
    draw_shape_lines_range(np_shape,image,Nose_Bridge_points)
    draw_shape_lines_range(np_shape,image,Lower_nose_points)
    draw_shape_lines_range(np_shape,image,Right_Eye_points,True)
    draw_shape_lines_range(np_shape,image,Left_Eye_points,True)
    draw_shape_lines_range(np_shape,image,Mouth_Outline_points,True)
    draw_shape_lines_range(np_shape,image,Mouth_Inner_points,True)

def shape_to_np(dlib_shape,dtype='int'):
    coordinate = np.zeros((dlib_shape.num_parts,2),dtype=dtype)
    for i in range(0,dlib_shape.num_parts):
        coordinate[i] = (dlib_shape.part(i).x,dlib_shape.part(i).y)
    
    return coordinate

p = "/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_3/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

test_image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_2/face_test.png')
gray_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)

image_1 = test_image.copy()
image_2 = test_image.copy()
image_3 = test_image.copy()
image_4 = test_image.copy()


rects = detector(gray_image,0)
for (i,rect) in enumerate(rects):
    shape = predictor(gray_image,rect)
    shape = shape_to_np(shape)

    draw_shape_line_all(shape,cv2.rectangle(image_1,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(0,255,0),1))
    
    draw_shape_points_pos_range(shape,cv2.rectangle(image_2,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(0,255,0),1),All_points)
    
    draw_shape_points_range(shape,cv2.rectangle(image_3,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(0,255,0),1),Jawline_points)
    
    draw_shape_points_pos_range(shape,cv2.rectangle(image_4,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(0,255,0),1),Left_Eye_points+Right_Eye_points+Nose_Bridge_points)



fig = plt.figure(figsize=(10,6))
fig.patch.set_facecolor('silver')
plt.suptitle("Landmarks detection using dlib",fontsize=14,fontweight='bold')

plt.subplot(221)
plt.imshow(image_1[:,:,::-1])
plt.axis('off')
plt.subplot(222)
plt.imshow(image_2[:,:,::-1])
plt.axis('off')
plt.subplot(223)
plt.imshow(image_3[:,:,::-1])
plt.axis('off')
plt.subplot(224)
plt.imshow(image_4[:,:,::-1])
plt.axis('off')
plt.show()

#%%

b = "C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/shape_predictor_5_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(b)

image_0 = test_image.copy()
rects = detector(gray_image,0)
for (i,rect) in enumerate(rects):
    shape = predictor(gray_image,rect)
    shape = shape_to_np(shape)
    
    draw_shape_points(shape,cv2.rectangle(image_0,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(0,255,0),1))

fig = plt.figure(figsize=(10,6))
fig.patch.set_facecolor('silver')
plt.suptitle("Landmarks detection using dlib",fontsize=14,fontweight='bold')

plt.imshow(image_0[:,:,::-1])
plt.title('Dlib using shape_predictor_5_face_landmarkes')
plt.axis('off')
plt.show()

#%%
# Deteccion de rasgos  faciales con face_recognition

# Deteccion de 68 marcas faciales
face_test = cv2.imread('/home/mbi/Documentos/Scripts/Mastering_OpenCV4/Seccion_2/face_test.png')
face_1 = face_test.copy()
face_2 = face_test.copy()

face_landmark_68 = face_recognition.face_landmarks(face_test)
print(face_landmark_68,"\n")

for face_landmark in face_landmark_68:
    for facial_feature in face_landmark.keys():
        for p in face_landmark[facial_feature]:
            cv2.circle(face_1,p,2,(0,255,0),-1)

face_landmark_5 = face_recognition.face_landmarks(face_test,model="small")
print(face_landmark_5,"\n")

for face_landmark in face_landmark_5:
    for face_feature in face_landmark.keys():
        for p in face_landmark[face_feature]:
            cv2.circle(face_2,p,2,(0,255,0),-1)
            
fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('silver')
plt.suptitle("Facial landmarks detection using  face_recognition",fontsize=14,fontweight='bold')

plt.subplot(121)
plt.imshow(face_1[:,:,::-1])
plt.title("68 facial landmarks")
plt.axis('off')
plt.subplot(122)
plt.imshow(face_2[:,:,::-1])
plt.title("5 facial landmarks")
plt.axis('off')
plt.show()


#%%
