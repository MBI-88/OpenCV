# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:00:12 2021

@author: MBI
"""
import cv2 
import matplotlib.pyplot as plt
import numpy as np
#%%
"""
Para descargar el modelo:
    
bvlc_alexnet.prototxt: https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/bvlc_alexnet.prototxt
bvlc_alexnet.caffemodel: http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel

"""
def show_imag(img,title,pos):
    img_rgb = img[:,:,::-1]
    plt.subplot(1,1,pos)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')

rows = open("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/synset_words.txt").read().strip().split("\n")
classes = [r[r.find(' ')+1:].split(',')[0] for r in rows]

net = cv2.dnn.readNetFromCaffe("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/bvlc_alexnet.prototxt","C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/bvlc_alexnet.caffemodel")

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/church.jpg')

blob = cv2.dnn.blobFromImage(image,1,(227,227),[104.,117.,123.])
print("Shape of blob: {}\n".format(blob.shape))

net.setInput(blob)
preds = net.forward()

t,_ = net.getPerfProfile()
print("Inference time: %.2f ms"%(t * 100.0  / cv2.getTickFrequency()),"\n")

indexes = np.argsort(preds[0])[::-1][:10]

text = "label: {}\nprobability: {:.2f}%".format(classes[indexes[0]],preds[0][indexes[0]] * 100)
y0,dy = 30,30

for i,line in enumerate(text.split('\n')):
    y = y0 + i * dy
    cv2.putText(image,line,(5,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

for (index,idx) in enumerate(indexes):
    print("{}. label: {}, probability: {:.3}".format(index + 1,classes[idx],preds[0][idx]))

fig = plt.figure(figsize=(10,6))
plt.suptitle("Image classification with OpenCV using AlexNet and caffe pre-trainded models",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

show_imag(image,"AlexNet and caffe pre-trained models",1)
plt.show()

#%%
"""
bvlc_googlenet.prototxt: https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/bvlc_googlenet.prototxt
bvlc_googlenet.caffemodel: http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
"""
image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/church.jpg')

googlenet = cv2.dnn.readNetFromCaffe("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/bvlc_googlenet.prototxt","C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/bvlc_googlenet.caffemodel")
blob = cv2.dnn.blobFromImage(image,1,(227,227),[104.,117.,123.])

googlenet.setInput(blob)
predict = googlenet.forward()
print("Shape of predict: {}\n".format(predict.shape))

t,_ = googlenet.getPerfProfile()
print("Inference time: %.2f ms"%(t * 100.0  / cv2.getTickFrequency()),"\n")

indexes = np.argsort(predict[0])[::-1][:10]

text = "label: {}\nprobability: {:.2f}%".format(classes[indexes[0]],predict[0][indexes[0]] * 100)
y0,dy = 30,30

for i,line in enumerate(text.split('\n')):
    y = y0 + i * dy
    cv2.putText(image,line,(5,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

for (index,idx) in enumerate(indexes):
    print("{}. label: {}, probability: {:.3}".format(index + 1,classes[idx],predict[0][idx]))

fig = plt.figure(figsize=(10,6))
plt.suptitle("Image classification with OpenCV using GoogleNet and caffe pre-trainded models",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

show_imag(image,"GoogleNet and caffe pre-trained models",1)
plt.show()

#%%
"""
(SqueezeNet v1.1 has 2.4x less computation than v1.0, without sacrificing accuracy.)

deploy.prototxt:
 https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/deploy.prototxt
squeezenet_v1.1.caffemodel:
 https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
"""
image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/church.jpg')

squeezenet = cv2.dnn.readNetFromCaffe("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/deploy_squeeze.prototxt","C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/squeezenet_v1.1.caffemodel")
blob = cv2.dnn.blobFromImage(image,1,(227,227),[104,117,123])

squeezenet.setInput(blob)
predicts = squeezenet.forward()
print("Shape of predicts: {}\n".format(predicts.shape))
predicts = predicts.reshape((1,len(classes)))

t,_ = squeezenet.getPerfProfile()
print("Inference time: %.2f ms"%(t * 100.0  / cv2.getTickFrequency()),"\n")

indexes = np.argsort(predicts[0])[::-1][:10]

text = "label: {}\nprobability: {:.2f}%".format(classes[indexes[0]],predicts[0][indexes[0]] * 100)
y0,dy = 30,30

for i,line in enumerate(text.split('\n')):
    y = y0 + i * dy
    cv2.putText(image,line,(5,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

for (index,idx) in enumerate(indexes):
    print("{}. label: {}, probability: {:.3}".format(index + 1,classes[idx],predicts[0][idx]))

fig = plt.figure(figsize=(10,6))
plt.suptitle("Image classification with OpenCV using SqueezeNet and caffe pre-trainded models",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

show_imag(image,"SqueezeNet and caffe pre-trained models",1)
plt.show()
#%%
"""
ResNet-50-deploy.prototxt:
 https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777
ResNet-50-model.caffemodel:
 https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777
"""

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/church.jpg')

resnet = cv2.dnn.readNetFromCaffe("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/ResNet-50-deploy.prototxt","C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/ResNet-50-model.caffemodel")

blob = cv2.dnn.blobFromImage(image,1,(224,224),[104,117,123])

resnet.setInput(blob)
predicts = resnet.forward()
print("Shape of predicts: {}\n".format(predicts.shape))

t,_ = resnet.getPerfProfile()
print("Inference time: %.2f ms"%(t * 100.0  / cv2.getTickFrequency()),"\n")

indexes = np.argsort(predicts[0])[::-1][:10]

text = "label: {}\nprobability: {:.2f}%".format(classes[indexes[0]],predicts[0][indexes[0]] * 100)

y0,dy = 30,30

for i,line in enumerate(text.split('\n')):
    y = y0 + i * dy
    cv2.putText(image,line,(5,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

for (index,idx) in enumerate(indexes):
    print("{}. label: {}, probability: {:.3}".format(index + 1,classes[idx],predicts[0][idx]))

fig = plt.figure(figsize=(10,6))
plt.suptitle("Image classification with OpenCV using ResNet and caffe pre-trainded models",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

show_imag(image,"RestNet and caffe pre-trained models",1)
plt.show()

#%%
"""
MobileNetSSD_deploy.caffemodel:
 https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc
MobileNetSSD_deploy.prototxt
 https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/daef68a6c2f5fbb8c88404266aa28180646d17e0/MobileNetSSD_deploy.prototxt
"""
image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/object_detection_test_image.png')

mobileNet = cv2.dnn.readNetFromCaffe("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/MobileNetSSD_deploy.prototxt","C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/MobileNetSSD_deploy.caffemodel")

blob = cv2.dnn.blobFromImage(image,0.007843,(300,300),[127.5,127.5,127.5])
print("Shape of blob: {}\n".format(blob.shape))

class_names = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car',8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike',15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20:'tvmonitor'}

mobileNet.setInput(blob)
detections = mobileNet.forward()

t,_ = mobileNet.getPerfProfile()
print("Inference time: {:.2f}ms\n".format(t * 1000.0 / cv2.getTickFrequency()))

dim = 300
for i in range(detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > 0.1:
        class_id = int(detections[0,0,i,1])
        
        xleftBottom = int(detections[0,0,i,3] * dim)
        yleftBottom = int(detections[0,0,i,4] * dim)
        xrightTop = int(detections[0,0,i,5] * dim)
        yrightTop = int(detections[0,0,i,6] * dim)
        
        heightFactor = image.shape[0]/dim
        widthFactor = image.shape[1]/dim 
        
        xleftBottom = int(widthFactor * xleftBottom)
        yleftBottom = int(heightFactor * yleftBottom)
        xrightTop = int(widthFactor * xrightTop)
        yrightTop = int(heightFactor * yrightTop)
        cv2.rectangle(image,(xleftBottom,yleftBottom),(xrightTop,yrightTop),(0,255,0),2)
        
        if class_id in class_names:
            label = class_names[class_id] + ": "+str(round(confidence * 100,2))
            labelSize,baseline = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,1,2)
            yleftBottom = max(yleftBottom,labelSize[1])
            cv2.rectangle(image,(xleftBottom,yleftBottom - labelSize[1]),(xleftBottom + labelSize[0],yleftBottom ),(0,255,0),cv2.FILLED)
            cv2.putText(image,label,(xleftBottom,yleftBottom),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

fig = plt.figure(figsize=(10,6))
plt.suptitle("Object detection with OpenCV using MobileNet-SSD and caffe pre-trainded models",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

show_imag(image,"MobileNet-SSD and caffe pre-trained models",1)
plt.show()


#%%

"""
(YOLOv3: An Incremental Improvement: https://pjreddie.com/media/files/papers/YOLOv3.pdf)
(yolov3.weights is not included as  exceeds GitHub's file size limit of 100.00 MB)
yolov3.weights: https://pjreddie.com/media/files/yolov3.weights
yolov3.cfg: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
"""

image = cv2.imread('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/object_detection_test_image.png')
(H,W) = image.shape[:2]
class_names = open("C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/coco.names").read().strip().split("\n")

yolov3 = cv2.dnn.readNetFromDarknet('C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/yolov3.cfg',"C:/Users/MBI/Documents/Python_Scripts/Mastering_OpenCV4/Seccion_3/yolov3.weights")

layer_names = yolov3.getLayerNames()
layer_names = [layer_names[i[0]-1] for i in yolov3.getUnconnectedOutLayers()]

blob  = cv2.dnn.blobFromImage(image,1/255.0,(416,416),swapRB=True,crop=False)
print("Shape of blob: {}\n".format(blob.shape))

yolov3.setInput(blob)
layerOutputs = yolov3.forward(layer_names)

t,_ = yolov3.getPerfProfile()
print("Inference time: {:.2f}ms\n".format(t * 1000.0 / cv2.getTickFrequency()))

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.25:
            box = detection[0:4] * np.array([W,H,W,H])
            (centerX,centerY,width,height) = box.astype('int')
            
            x = int(centerX -(width/2))
            y = int(centerY - (height/2))
            boxes.append([x,y,int(width),int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
        
indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.3)
print("Shape of indices: {}\n".format(indices.shape))

if len(indices) > 0:
    for i in indices.flatten():
        (x,y) = (boxes[i][0],boxes[i][1])
        (w,h) = (boxes[i][2],boxes[i][3])
        
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        label = "{} {:.4f}".format(class_names[class_ids[i]],confidences[i] * 100)
        labelSize,baseline = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,1,2)
        y = max(y,labelSize[1])
        cv2.rectangle(image,(x,y-labelSize[1]),(x+labelSize[0],y),(0,255,0),cv2.FILLED)
        cv2.putText(image,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)


fig = plt.figure(figsize=(14,8))
plt.suptitle("Object detection with OpenCV using YOLOV3 and caffe pre-trainded models",fontsize=14,fontweight='bold')
fig.patch.set_facecolor('silver')

show_imag(image,"YOLOV3 and caffe pre-trained models",1)
plt.show()

#%%


