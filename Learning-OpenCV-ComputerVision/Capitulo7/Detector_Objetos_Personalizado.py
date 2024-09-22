# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:44:46 2021

@author: MBI
"""
import cv2,pathlib,os
import numpy as np
#%%
# Detectando personas con descriptores HOG

def is_inside(i,o):
    ix,iy,iw,ih = i
    ox,oy,ow,oh = o
    return ix > ox and ix + iw < ox + ow and iy > oy and iy + ih < oy + oh

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
img = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo7/Learning OpenCV 4 Computer Vision with Python 3_page364_image105.jpg")

found_rects,found_weights = hog.detectMultiScale(img,winStride=(4,4),scale=1.02,finalThreshold=1.9)

found_rects_filtered,found_weights_filtered = [],[]

for ri,r in enumerate(found_rects):
    for qi,q in enumerate(found_rects):
        if ri != qi and is_inside(r,q):
            break
        else:
            found_rects_filtered.append(r)
            found_weights_filtered.append(found_weights[ri])

for ri,r in enumerate(found_rects_filtered):
    x,y,w,h = r
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    text = "%.2f" % found_weights_filtered[ri]
    cv2.putText(img,text,(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

cv2.imshow("Detection",img)

while (cv2.waitKey(1) == -1):
    continue
cv2.destroyWindow("Detection")
#%%
# Creando y entrenando un detector de objetos

BOW_NUM_TRAINING_SAMPLE_PER_CLASS = 10
SVM_NUM_TRAINING_SAMPLE_PER_CLASS = 100
pos_car = []
neg_car = []
train_label = []
train_data = []

path = "C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo7/CarData"
dir_path = pathlib.Path(path)
name = ["Car","NoCar"]

sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE,tree=5)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params,search_params)
bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
bow_extractor = cv2.BOWImgDescriptorExtractor(sift,flann)

for f in ["Car","NoCar"]:
    if f == "Car":
        path_dir = os.path.join(dir_path,f)
        for x in os.listdir(path_dir):
            pos_car.append(path_dir+"/"+x)
            
    else:
        path_dir = os.path.join(dir_path,f)
        for x in os.listdir(path_dir):
            neg_car.append(path_dir+"/"+x)
            

def add_sample(paths):
    img = cv2.imread(paths,cv2.IMREAD_GRAYSCALE)
    keypoint,descriptor = sift.detectAndCompute(img,None)
    if descriptor is not None:
        bow_kmeans_trainer.add(descriptor)

for i in range(BOW_NUM_TRAINING_SAMPLE_PER_CLASS):
    add_sample(pos_car[i])
    add_sample(neg_car[i])

voc = bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc)

def extract_bow_descripors(img):
    features = sift.detect(img)
    return bow_extractor.compute(img,features)

for i in range(SVM_NUM_TRAINING_SAMPLE_PER_CLASS):
    pos_img = cv2.imread(pos_car[i],cv2.IMREAD_GRAYSCALE)
    pos_descriptors = extract_bow_descripors(pos_img)
    if pos_descriptors is not None:
        train_data.extend(pos_descriptors)
        train_label.append(1)
    neg_img = cv2.imread(neg_car[i],cv2.IMREAD_GRAYSCALE)
    neg_descriptors = extract_bow_descripors(neg_img)
    if neg_descriptors is not None:
        train_data.extend(neg_descriptors)
        train_label.append(-1)

svm = cv2.ml.SVM_create()
svm.train(np.array(train_data),cv2.ml.ROW_SAMPLE,np.array(train_label))

    

#%%
# Test

for test_img in ["C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo7/Test/150304410.jpg","C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo7/CarData/NoCar/Learning OpenCV 4 Computer Vision with Python 3_page364_image1.jpg"]:
    img = cv2.imread(test_img)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    descriptors = extract_bow_descripors(gray_img)
    prediction = svm.predict(descriptors)
    if prediction[1][0][0] == 1.0:
        text = "car"
        color = (0,255,0)
    else:
        text = "not car"
        color = (0,0,255)
    cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
    cv2.imshow(test_img,img)

while (cv2.waitKey(1) == -1):
    continue
cv2.destroyAllWindows()
        
# Nota: El set de entrenamiento no es el correcto
#%%
# Combinando SVM con ventana deslizante

def non_max_suppression_fast(boxes, overlapThresh):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []

  # initialize the list of picked indexes 
  pick = []

  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
  scores = boxes[:,4]
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the score/probability of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(scores)[::-1]

  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]

    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > overlapThresh)[0])))

  # return only the bounding boxes that were picked
  return boxes[pick]

SVM_SCORE_THRESHOLD = 1.85
NMS_OVERlAP_THRESHOLD = 0.15

bow_kmeans_trainer = cv2.BOWKMeansTrainer(12)

svm_ = cv2.ml.SVM_create()
svm_.setType(cv2.ml.SVM_C_SVC)
svm_.setC(25)
svm_.train(np.array(train_data),cv2.ml.ROW_SAMPLE,np.array(train_label))

def pyramid(img,scale_factor=1.25,min_size=(200,80),max_size=(600,600)):
    h,w = img.shape
    min_w,min_h = min_size
    max_w,max_h = max_size
    while w >= min_w and h >= min_h:
        if w <= max_w and h <= max_h:
            yield img
        w /= scale_factor
        h /= scale_factor
        img = cv2.resize(img,(int(w),int(h)),interpolation=cv2.INTER_AREA)

def sliding_windows(img,step=20,window_size=(100,100)):
    img_h,img_w = img.shape
    window_w,window_h = window_size
    for y in range(0,img_w,step):
        for x in range(0,img_h,step):
            roi = img[y:y+window_h,x:x+window_w]
            roi_h,roi_w = roi.shape
            if roi_w == window_w and roi_h == window_h:
                yield (x,y,roi)
    

for test_img in ["C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo7/Test/150304421.jpg","C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo7/Test/150304453.jpg","C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo7/Test/Learning OpenCV 4 Computer Vision with Python 3_page364_image55.jpg","C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo7/Learning OpenCV 4 Computer Vision with Python 3_page364_image105.jpg"]:
    img = cv2.imread(test_img)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pos_rets = []
    for resized in pyramid(gray_img):
        for x,y,roi in sliding_windows(resized):
            descriptors = extract_bow_descripors(roi)
            if descriptors is None:
                continue
            prediction = svm_.predict(descriptors)
            if prediction[1][0][0] == 1.0:
                raw_prediction = svm_.predict(descriptors,flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            score = -raw_prediction[1][0][0]
            if score > SVM_SCORE_THRESHOLD:
                h,w = roi.shape
                scale = gray_img.shape[0] / float(resized.shape[0])
                pos_rets.append([int(x*scale),int(y*scale),int((x+w)*scale),int((y+h)*scale),score])

    pos_rets = non_max_suppression_fast(np.array(pos_rets),NMS_OVERlAP_THRESHOLD)

    for x0,y0,x1,y1,score in pos_rets:
        cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),(0,255,255),2)
        text = "%.2f" % score
        cv2.putText(img,text,(int(x0),int(y0)-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imshow(test_img,img)

while (cv2.waitKey(1) == -1):
    continue
cv2.destroyAllWindows()
#%%
# Salvando y cargando el modelo SVM
# svm.save('my_svm.xml') / svm.load('my_svm.xml')


