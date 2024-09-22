#%% Teoria
"""
Eliginedo el tamaño de las capas de entrada segun conceptos:

    .Si el tamaño  de la capa de entrada es grande entonces el numero de neuronas de las capas ocultas debe
    estar entre el tamaño de entrada y el de salida, como norma mas cercano al de salida.
    .Si el tamaño de al entrada y de la salida es pequeño entonces el tamaño de las capas ocultas debe ser
    el mas grande.
    .Si el tamño de la capa de entrada  es pequeño pero el de la salida es grande entonces las capas ocultas deben
    ser cercanas al tamaño de la de entrada.
"""
#%% Bibliotecas
import cv2 
import numpy as np 
from random import randint,uniform
import gzip,pickle
#%% Entrenando una red neuronal basica

ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([9,15,9],np.uint8))

ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM,0.6,1.0)
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP,0.1,0.1)
ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,100,1.0))

train = np.array([[1.2,1.3,1.9,2.2,2.3,2.9,3.0,3.2,3.3]],np.float32)
layout = cv2.ml.ROW_SAMPLE
training_response = np.array([[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]],np.float32)
data = cv2.ml.TrainData_create(train, layout, training_response)

ann.train(data)


test_sample = np.array([[1.4,1.5,1.2,2.0,2.5,2.8,3.0,3.1,3.8]],np.float32)
prediction = ann.predict(test_sample)
print(prediction)
#%% Entrenamiento en multiples epocas

animals_net = cv2.ml.ANN_MLP_create()
animals_net.setLayerSizes(np.array([3,50,4],np.uint8))
animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM,0.2,1.0)
animals_net.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP,0.1,0.1)
animals_net.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,100,1.0))

# Clases
def dog_sample():
    return [uniform(10.0, 20.0),uniform(1.0,1.5),randint(38,42)]

def dog_class():
    return [1,0,0,0]

def condor_sample():
    return [uniform(3.0, 10.0),randint(3.0,5.0),0]

def condor_class():
    return [0,1,0,0]

def dolphin_sample():
    return [uniform(30.0, 190.0),uniform(5.0, 15.0),randint(80, 100)]

def dolphin_class():
    return [0,0,1,0]

def dragon_sample():
    return [uniform(1200.0, 1800.0),uniform(30.0, 40.0),randint(160, 180)]

def dragon_class():
    return [0,0,0,1]

def  record(sample,classification):
    return (np.array([sample],np.float32),np.array([classification],np.float32))

RECORDS = 20000
records = []

for x in range(0,RECORDS):
    records.append(record(dog_sample(),dog_class()))
    records.append(record(condor_sample(),condor_class()))
    records.append(record(dolphin_sample(),dolphin_class()))
    records.append(record(dragon_sample(),dragon_class()))

EPOCHS = 5
for e in range(0,EPOCHS):
    print("epoch: %d" % e)
    for t,c in records:
        data = cv2.ml.TrainData_create(t, cv2.ml.ROW_SAMPLE,c)
        if animals_net.isTrained():
            animals_net.train(data,cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
        
        else:
            animals_net.train(data,cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
            

TESTS = 100
dog_results = 0
for x in range(0,TESTS):
    clas = int(animals_net.predict(np.array([dog_sample()],np.float32))[0])
    print("class: %d" % clas)
    if clas == 0:
        dog_results += 1

condor_results = 0
for x in range(0,TESTS):
    clas = int(animals_net.predict(np.array([condor_sample()],np.float32))[0])
    print("class: %d" % clas)
    if  clas == 1 :
        condor_results += 1
        
                   
dolphin_results = 0
for x in range(0,TESTS):
    clas = int(animals_net.predict(np.array([dolphin_sample()],np.float32))[0])
    print("class: %d" % clas)
    if  clas == 2 :
        dolphin_results += 1

dragon_results = 0
for x in range(0,TESTS):
    clas = int(animals_net.predict(np.array([dragon_sample()],np.float32))[0])
    print("class: %d" % clas)
    if  clas == 3 :
        dragon_results += 1


print("dog accuracy : %.2f%%" % (100.0 * dog_results/TESTS))
print("condor accuracy : %.2f%%" % (100.0 * condor_results/TESTS))
print("dolphin accuracy : %.2f%%" % (100.0 * dolphin_results/TESTS))
print("dragon accuracy : %.2f%%" % (100.0 * dragon_results/TESTS))

#%% Entrenando con MININST DATASET

def load_data():
    mnist = gzip.open("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/mnist.pkl.gz",'rb')
    training_data,test_data = pickle.load(mnist)
    mnist.close()
    return (training_data,test_data)

def vertorized_result(y):
    e = np.zeros((10,),np.float32)
    e[y] = 1.0
    return e

def wrap_data():
    tr_d,te_d = load_data()
    training_inputs = tr_d[0]
    training_results = [vertorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs,training_results)
    test_data = zip(te_d[0],te_d[1])
    return (training_data,test_data)

def create_ann(hidden_node=60):
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([784,hidden_node,10]))
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM,0.5,1.0)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP,0.1,0.1)
    ann.setTermCriteria((cv2.TermCriteria_MAX_ITER | cv2.TERM_CRITERIA_EPS,100,1.0))
    return ann

def train(ann,samples=50000,epochs=10):
    tr,test = wrap_data()
    tr = list(tr)
    
    for epoch in range(epochs):
        print("Completed {}/{} epochs".format(epoch, epochs))
        counter = 0
        for img in tr:
            if (counter > samples):
                break
            if (counter % 1000 == 0):
                print("Epoch {}: Trained on {}/{} samples".format(epoch,counter,samples))
            counter += 1
            sample,response = img
            data = cv2.ml.TrainData_create(np.array([sample],np.float32),cv2.ml.ROW_SAMPLE,np.array([response],np.float32))
            if ann.isTrained():
                ann.train(data,cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
            else:
                ann.train(data,cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
            
    
    print("Completed all epochs")
    return ann,test

def predict(ann,sample):
    if sample.shape != (784,):
        if sample.shape != (28,28):
            sample = cv2.resize(sample,(28,28),interpolation=cv2.INTER_LINEAR)
        
        sample = sample.reshape(784,)
    return ann.predict(np.array([sample],np.float32))

def test(ann,test_data):
    num_tests = 0
    num_correct = 0
    for img in test_data:
        num_tests += 1
        sample,correct_digit_class = img
        digit_class = predict(ann, sample)[0]
        if digit_class == correct_digit_class:
            num_correct += 1
    
    print("Accuracy: {}%".format(100.0 * num_correct/num_tests))


ann,test_data = train(create_ann())
test(ann,test_data)
#%% Modulo principal

def insider(r1,r2):
    x1,y1,w1,h1 = r1
    x2,y2,w2,h2 = r2
    return (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and (y1+h1 < y2+h2)

def wrap_digit(rect,img_w,img_h):
    x,y,w,h = rect
    
    x_center = x + w//2
    y_center = y + h//2
    
    if (h > w):
        w = h
        x  = x_center - (w//2)
    else:
        h = w
        y = y_center - (h//2)
    
    padding = 5
    x -= padding
    y -= padding
    w +=  2 * padding
    h += 2 * padding
    
    if x < 0:
        x = 0
    elif x > img_w:
        x = img_w
    if y < 0:
        y = 0
    elif y > img_h:
        y = img_h
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y
    
    return x,y,w,h

img_path = cv2.imread("C:/Users/MBI/Documents/Python_Scripts/Learning-OpenCV-ComputerVision/Capitulo10/digits_0.jpg",cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img_path,cv2.COLOR_BGR2GRAY)
cv2.GaussianBlur(gray, (7,7),0,gray)

ret,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
erode_kernel = np.ones((2,2),np.uint8)
thresh = cv2.erode(thresh, erode_kernel,thresh,iterations=2)

contours,hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

rectangle = []
img_h,img_w = img_path.shape[:2]
img_area = img_w * img_h

for c in contours:
    a = cv2.contourArea(c)
    if a >= 0.98 * img_area or a <= 0.0001 * img_area:
        continue
    
    r = cv2.boundingRect(c)
    is_inside = False
    for q in rectangle:
        if insider(r, q):
            is_inside = True
            break
    if not is_inside:
        rectangle.append(r)
        
for r in rectangle:
    x,y,w,h = wrap_digit(r, img_w, img_h)
    roi = thresh[y:y+h,x:x+w]
    digit_class = int(predict(ann, roi)[0])
    
    cv2.rectangle(img_path, (x,y), (x+w,y+h), (0,255,0),2)
    cv2.putText(img_path, "%d"% digit_class, (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2)
    

cv2.imshow("thresh", thresh)
cv2.imshow("Detected", img_path)

while (cv2.waitKey(1) != ord('q')):
    continue

cv2.destroyAllWindows()
    

# Para salvar la red usar :
    # ann.save('my_ann.xml')
# Para cargar la red usar:
  # ann = cv2.ml.ANN_MLP_create()
  # ann.load('my_ann.xml')  

#%% Encongrando otras aplicaciones potenciales
