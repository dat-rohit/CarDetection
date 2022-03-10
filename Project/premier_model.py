#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Install this at the beginning
#get_ipython().system('pip install python-resize-image')
#get_ipython().system('pip install opencv-python')


# ### Hyperparameters to use and Pre Defined parameters

# In[1]:


# IoU Threshold 0.6
iou_thresh = 0.6

# Objectness Threshold 0.6
obj_thresh = 0.6

# network's input dimensions
net_h, net_w = 416, 416

# 3 anchor boxes (width,height) pairs
anchors = [ [[116, 90], [156, 198], [373, 326]], 
             [[30, 61], [62, 45], [59, 119]], 
             [[10, 13], [16, 30], [33, 23]]]


NUM_CLASS = 80


# ### Pre Defined

# ### Import Model Yolov3

# In case the yolov3.weights file does not exist, it can be downloaded directly from the DarkNet website:

# In[ ]:


#get_ipython().system('wget https://pjreddie.com/media/files/yolov3.weights #needed to run')


# After the pretrained weights are downloaded they can be loaded into the Keras model using the following:

# In[ ]:


from dis import dis
import os
from unicodedata import name
import streamlit as st
import math
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob
import matplotlib.pyplot as plt
from resizeimage import resizeimage
import cv2
from Utils import preprocess_image
import random
from random import randrange
from Model import extract_features
import tensorflow as tf


# In[ ]:


from Model import YOLOv3
from tensorflow.keras.layers import Input
from WeightsReader import WeightReader

# create Yolo model
model = YOLOv3(Input(shape=(None, None, 3)), NUM_CLASS);    
#model.summary()

# load the weights trained on COCO into the model
WeightReader("./yolov3.weights").load_weights(model)


# ### All the importations

# # Functions we will need in this notebook

# In[ ]:




labels = ["person",        "bicycle",       "car",          "motorbike",      "aeroplane",     "bus",        "train",   "truck",           "boat",          "traffic light", "fire hydrant", "stop sign",      "parking meter", "bench",           "bird",          "cat",           "dog", "horse", "sheep",          "cow",           "elephant",   "bear",    "zebra", "giraffe",           "backpack",      "umbrella",      "handbag",      "tie",            "suitcase",      "frisbee",    "skis",    "snowboard",           "sports ball",   "kite",          "baseball bat", "baseball glove", "skateboard",    "surfboard",           "tennis racket", "bottle",        "wine glass",   "cup",            "fork",          "knife",      "spoon",   "bowl", "banana",           "apple",         "sandwich",      "orange",       "broccoli",       "carrot",        "hot dog",    "pizza",   "donut", "cake",           "chair",         "sofa",          "pottedplant",  "bed",            "diningtable",   "toilet",     "tvmonitor", "laptop", "mouse",           "remote",        "keyboard",      "cell phone",   "microwave",      "oven",          "toaster",    "sink",    "refrigerator",           "book",          "clock",         "vase",         "scissors",       "teddy bear",    "hair drier", "toothbrush"]

def preprocess_image(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)/new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)/new_h
        new_h = net_h    
    
    # resize the image to the new size. Normalize the data and reflect the pixels [:,:,::-1]
    resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), 
              int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def getScaleFactors(image_w, image_h, net_w, net_h):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_h = (image_h*net_w)/image_w
        new_w = net_w
    else:
        new_w = (image_w*net_h)/image_h
        new_h = net_h    
    
    x_scale = float(new_w)/net_w
    y_scale = float(new_h)/net_h
    x_offset = (net_w - new_w)/2./net_w
    y_offset = (net_h - new_h)/2./net_h
    
    return x_scale, y_scale, x_offset, y_offset
    

def draw_boxes(image, boxes, classes, scores, image_w, image_h, net_w, net_h):
    
    x = boxes[:, 0].numpy()
    y = boxes[:, 1].numpy()
    w = boxes[:, 2].numpy()
    h = boxes[:, 3].numpy()

    x2 = x+w
    y2 = y+h

    x_scale, y_scale, x_offset, y_offset = getScaleFactors(image_w, image_h, net_w, net_h)
    
    x  = (x  - x_offset) / x_scale * image_w 
    x2 = (x2 - x_offset) / x_scale * image_w 
    y  = (y  - y_offset) / y_scale * image_h
    y2 = (y2 - y_offset) / y_scale * image_h 

    
    start = list(zip(x.astype(int), y.astype(int) ))
    end = list(zip( x2.astype(int), y2.astype(int) ))
    list_start=[]
    list_end=[]
    list_label=[]
    for i in range(len(boxes)):        
        r = randrange(255)
        g = randrange(255)
        b = randrange(255)                            
        color = (r,g,b)        
        label = np.argmax(classes[i,:], axis=0)
        if (labels[label]=="car" or labels[label]=="truck" or labels[label]=="bus"):
            #cv2.rectangle(image, start[i], end[i], color, 3) 
            proba_label = np.max(classes[i,:], axis=0)
            list_start.append(start[i])
            list_end.append(end[i])
            list_label.append(proba_label)
            #cv2.putText(image, labels"car", (start[i][0], start[i][1] - 10), 
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    #(0,0,0), 2)
    if len(list_start) == 0 :
        list_start.append((-math.pi,-math.pi))
        list_end.append((-math.pi,-math.pi))
        list_label.append((-math.pi,-math.pi))
    return image, list_start, list_end, list_label    

def ApplyModel(output):
    boxes = np.empty([1, 4])
    scores = np.empty([1, ])
    classes = np.empty([1, 80])

    for i in range(len(output)):    
        _, S = output[i].shape[:2]
    
        b, s, c = extract_features( output[i], anchors[i], S, N=3, num_classes=(80), net_wh=(416,416))
    
        boxes = np.concatenate((boxes, b), axis=0)
        scores = np.concatenate((scores, s), axis=0)
        classes = np.concatenate((classes, c), axis=0)



    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                                        boxes, scores, len(boxes), 
                                        iou_threshold=iou_thresh, 
                                        score_threshold=obj_thresh ,
                                        soft_nms_sigma=0.6)

    selected_boxes = tf.gather(boxes, selected_indices)
    selected_classes = tf.gather(classes, selected_indices)
    
    #A adapter pour pouvoir avoir qu'une fonction excécutant notre model
      #'''  image', list_start, list_end, list_label '= draw_boxes(X[k], selected_boxes,selected_classes, selected_scores, image_w, image_h, net_w, net_h)
    #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8'))
    '''if len(list_label) > 1 :
        maxi = np.argmax(list_label)
        start = list_start[maxi]
        end = list_end[maxi]
        case_coordonnee = np.array([start[0], end[0], start[1], end[1]])
    if (list_start[0][0]!=-math.pi) and (list_start[0][1]!=-math.pi):
        X_new.append(X[k])
        y_test.append(case_coordonnee) #: à utiliser dans le cas général 
        y_valid.append(y[k]) #à utiliser pour fit le 2ème model 
        car_model_new.append(car_model[k])
        '''
        
    return selected_indices, selected_scores, selected_boxes, selected_classes

def create_dataset_for_second_model(X,y):
    X_new =[]
    y_valid = []
    y_test =[]
    car_model_new=[]
    for k in range(len(X)):
        input = preprocess_image(X[k], net_h, net_w)
        output = model.predict(input)
        selected_indices, selected_scores, selected_boxes, selected_classes = ApplyModel(output)
        image, list_start, list_end, list_label = draw_boxes(X[k], selected_boxes,selected_classes, selected_scores, image_w, image_h, net_w, net_h)
        #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8'))
        if (list_start[0][0]==-math.pi) and (list_start[0][1]==-math.pi):
            '''print('there is no car')'''
        else :
            maxi = np.argmax(list_label)
            list_start[0] = list_start[maxi]
            list_end[0] = list_end[maxi]
            #fig = plt.figure(figsize=(12, 12))
            #axis = fig.add_subplot()
            case_coordonnee = np.array([list_start[0][0], list_end[0][0], list_start[0][1], list_end[0][1]])
            #cv2.rectangle(image, list_start[0], list_end[0], color=(0, 255, 0))    #ajouter ,3     
            #cv2.putText(image, "car", (list_start[0][0], list_start[0][1] - 10), 
            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, #(0,0,0), 2)
            X_new.append(X[k])
            y_test.append(case_coordonnee)
            y_valid.append(y[k])
            
    X_new = np.array(X_new)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)
    car_model_new = np.array(car_model_new)
    
    return X_new, y_test, y_valid, car_model_new

def try_for_one_image(image):

    input = preprocess_image(a, net_h, net_w)
    output = model.predict(input)
    image_h, image_w, _ = a.shape
    selected_indices, selected_scores, selected_boxes, selected_classes = ApplyModel(output)
    image, list_start, list_end, list_label = draw_boxes(a, selected_boxes,selected_classes, selected_scores, image_w, image_h, net_w, net_h)
    if len(list_label) > 1 :
            maxi = np.argmax(list_label)
            list_start[0] = list_start[maxi]
            list_end[0] = list_end[maxi]
            fig = plt.figure(figsize=(12, 12))
            axis = fig.add_subplot()
            case_coordonnee = np.array([list_start[0][0], list_end[0][0], list_start[0][1], list_end[0][1]])
            cv2.rectangle(image, list_start[0], list_end[0], color=(0, 255, 0))    #ajouter ,3     
            cv2.putText(image, "car", (list_start[0][0], list_start[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0,0,0), 2)

    else :
        if (list_start[0][0]==-math.pi) and (list_start[0][1]==-math.pi):
            print('there is no car')
        else :
            fig = plt.figure(figsize=(12, 12))
            axis = fig.add_subplot()
            case_coordonnee = np.array([list_start[0][0], list_end[0][0], list_start[0][1], list_end[0][1]])
            cv2.rectangle(image, list_start[0], list_end[0], color=(0, 255, 0)) #ajouter ,3         
            cv2.putText(image, "car", (list_start[0][0], list_start[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0,0,0), 2)
    
    plt.imshow(delimited_image(image))   
    plt.show()
    return


def delimit_image(image):
    color_1 = np.array(image)
    color_1 = color_1.mean(axis=2)
    mask = color_1 < 255

    pos2 = 0
    for i in range(len(mask)):
        for j in range(len(mask[1])):
            if (mask[i,j] == True):
                pos2 = (i,j)
    pos1 = 0
    for i in range(len(mask)):
        for j in range(len(mask[1])):
            if (mask[i,j] == True) and (pos1==0):
                pos1 = (i,j)
    
    return image[pos1[0]:pos2[0]+1, pos1[1]:pos2[1]+1,: ]

def delimit_box(image_delimited):
    color_0 = np.array(image[:,:,0])
    mask0 = color_0 == 0 
    
    color_1 = np.array(image[:,:,1])
    mask1 = color_1 == 255
    
    color_2 = np.array(image[:,:,2])
    mask2 = color_2 == 0
    
    pos2 = 0
    for i in range(1, len(mask1)):
        for j in range(1, len(mask1[1])):
            if (mask1[i,j] == True) and (mask0[i,j] == True) and (mask2[i,j]==True):
                pos2 = (i,j)
    pos1 = 0
    for i in range(1, len(mask1)):
        for j in range(1, len(mask1[1])):
            if (mask1[i,j] == True) and (mask0[i,j] == True) and (mask2[i,j]==True) and (pos1==0):
                pos1 = (i,j)
    
    return pos1[0],pos2[0], pos1[1], pos2[1]

def convert_ratio(image_origine_shape, image):
    image = delimit_image(image)
    xmin, xmax, ymin, ymax = delimit_box(image)
    ymin_f = ymin * image_origine_shape[0]/image.shape[0]
    ymax_f = ymax * image_origine_shape[0]/image.shape[0]
    xmin_f = xmin * image_origine_shape[1]/image.shape[1]
    xmax_f = xmax * image_origine_shape[1]/image.shape[1]
    return [xmin_f, xmax_f, ymin_f, ymax_f]


# # Load Data for test and use on our second model

# In[ ]:


#Dataset : 2600 images création de X et y, que l'on va tester sur notre model préexistant
import cv2
from Utils import preprocess_image
'''
y_df=pd.read_csv('../../../datasets/datasets_train/train_annotation/_annotation.csv')
classes=y_df["class"]
directory=y_df["im_name"]
car_model = y_df["models"]
#X creation of this dataset 
X=[]
shape_origin=[]
img = ''
train_dir='../../../datasets/datasets_train/train/'
for k in range(len(classes)):
    img = Image.open(train_dir + directory[k])
    img = np.array(img, dtype=np.uint8)
    shape_origin.append(img.shape)
    img=Image.fromarray(img)
    img = resizeimage.resize_contain(img, [400, 250])
    val = np.array(img, dtype=np.uint8)
    X.append(val[:,:,[0,1,2]])
    
X = np.array(X)

#y = [[xmin, xmax, ymin, ymax],...]
x_min = np.array(y_df["x_min"])
x_max = np.array(y_df["x_max"])
y_min = np.array(y_df["y_min"])
y_max = np.array(y_df["y_max"])
y = []
for k in range(len(x_min)):
    y.append([x_min[k], x_max[k], y_min[k], y_max[k]])
y = np.array(y)

image_h, image_w, _ = X[0].shape 


# In[ ]:


#In X_newdataset there are only car images, y_newdataset there are model of car, and list of boxes on other list
X_newdataset, predict_boxes, real_boxes, y_newdataset = create_dataset_for_second_model(X,y)


# ## y_new -> coordinates of car on the dateset test 

# In[ ]:



import glob
x = glob.glob("../../../datasets/datasets_test/test/*")
X=[]

for k in range(len(x)):
    img = Image.open(x[k])
    img = resizeimage.resize_contain(img, [400, 250])
    val = np.array(img, dtype=np.uint8)
    X.append(val[:,:,[0,1,2]])
image_h, image_w, _ = X[0].shape     
X = np.array(X)

y_new =[]
for k in range(len(X)):
    input = preprocess_image(X[k], net_h, net_w)
    output = model.predict(input)
    selected_indices, selected_scores, selected_boxes, selected_classes = ApplyModel(output)
    image, list_start, list_end, list_label = draw_boxes(X[k], selected_boxes,selected_classes, selected_scores, image_w, image_h, net_w, net_h)
    if (list_start[0][0]==-math.pi) and (list_start[0][1]==-math.pi):
        y_new.append([0, 0, 0, 0]) #if there is no car 
    else :
        maxi = np.argmax(list_label)
        list_start[0] = list_start[maxi]
        list_end[0] = list_end[maxi]
        case_coordonnee = np.array([list_start[0][0], list_end[0][0], list_start[0][1], list_end[0][1]])
        y_new.append(case_coordonnee)
        
y_new = np.array(y_new)
X_new = x[0][85:]

image_h, image_w, _ = X[0].shape 

'''
# ## If you want to apply our algorithm on an image (to detect car)

# In[ ]:


#Ne pas run pour le moment  
#Pour afficher sur une image -> input en a le tableau numpy de l'image ! 
a ="image_file"
img = Image.open(x[k])
img = resizeimage.resize_contain(img, [400, 250])
val = np.array(img, dtype=np.uint8)
image = val[:,:,[0,1,2]]
try_for_one_image(image)


# In[ ]:


fig = plt.figure(figsize=(12, 12))
crop_img=b[case_coordonnee[2]+1:case_coordonnee[3],case_coordonnee[0]+1:case_coordonnee[1]]
crop_img=Image.fromarray(crop_img)
crop_img=resizeimage.resize_contain(crop_img,[400,400])
plt.imshow(crop_img)


# In[ ]:


print(image_final.shape)
photo = Image.open(train_dir + directory[111])
photo = np.array(photo, dtype=np.uint8)
photo.shape


# In[ ]:


#Test size, pas utile pour l'instant

photo = Image.open(train_dir + directory[374])
shape_max = np.array(photo, dtype=np.uint8).shape
image_h, image_w, _ = shape_max

photo = Image.open(train_dir + directory[0])
shape_init = np.array(photo, dtype=np.uint8).shape
photo1 = resizeimage.resize_contain(photo, [400, 250])
val = np.array(photo1, dtype=np.uint8)
photo1 = val[:,:,[0,1,2]]
shape_inter = photo1.shape
photo2 = preprocess_image(photo1, net_h, net_w)


photo2 = photo2.reshape((416,416,3))
print(shape_init, shape_final, photo2.shape)


# In[ ]:


#IoU Calcul of our algorithm 
IoU =[]

for i in range(len(y_valid)):
    x_inter1=max(y_valid[i,0], y_test[i,0])
    x_inter2=min(y_valid[i,1], y_test[i,1])
    y_inter1=max(y_valid[i,2], y_test[i,2])
    y_inter2=min(y_valid[i,3], y_test[i,3])

    width_inter = x_inter2 - x_inter1
    height_inter = y_inter2 - y_inter1

    area_inter = height_inter * width_inter

    width_box1= y_test[i,1] - y_test[i,0]
    height_box1 = y_test[i,3] - y_test[i,2]
    width_box2= y_valid[i,1] - y_valid[i,0]
    height_box2 = y_valid[i,3] - y_valid[i,2]

    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    area_union = area_box1 + area_box2 - area_inter
    IoU.append(area_inter/area_union)

print(y_valid[1377])
print(y_test[1377])
IoU = np.array(IoU)

for k in range(len(IoU)):
    if IoU[k]=='nan':
        i=k
i


# In[ ]:


#Rohit et Nour 
i=0
y_valid = np.array(y_valid, dtype=np.uint8)
cropped_images=[]
for k in range(len(X_new)):
    crop_img=X_new[k][y_test[i][2]:y_test[i][3],y_test[i][0]:y_test[i][1]]
    crop_img=Image.fromarray(crop_img)
    crop_img=resizeimage.resize_contain(crop_img,[400,400])
    val=np.array(crop_img,dtype=np.uint8)
    cropped_images.append(val[:,:,[0,1,2]])

    
cropped_images=np.array(cropped_images)
plt.imshow(cropped_images[2])    


# In[ ]:


plt.imshow(cropped_images[5])
car_model_new[5]


# In[ ]:


#pas utile 
print(cropped_images.shape)


img=X_new[0]
plt.imshow(img)
print(car_model_new.shape)
print(car_model_new[0])
img_crop=img[y_new[0][2]:y_new[0][3],y_new[0][0]:y_new[0][1]]
plt.imshow(Image.fromarray(img_crop))


# In[ ]:


X_1D=cropped_images.mean(axis=3)
plt.imshow(X_1D[0])

#PCA : réduire dimension marche pas, randomforest donne un score nul 

X_1D_bis=[]
for k in range(len(X_1D)):
    X_1D_bis.append(X_1D[k].ravel())
    
X_1D_bis=np.array(X_1D_bis)


# In[ ]:


#plt.hist(X_1D_bis[0])


# In[ ]:


#Loading all the models available 
#We have a 100 car models to deal with 
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

encoder = LabelEncoder()
#y = encoder.fit_transform(car_model_new)

X_train, X_test, y_train, y_test = train_test_split(X_1D_bis, car_model_new, test_size=0.2, random_state=10)
y_train_encode = encoder.fit_transform(y_train)
y_test_encode = encoder.transform(y_test)


# In[ ]:


#Setting up and testing the model 

knn = KNeighborsClassifier(n_neighbors=101)

knn.fit(X_train, y_train_encode)

knn.score(X_test, y_test_encode)


# In[ ]:


len(y_df["models"].unique())


# In[ ]:


x = np.array([12])
np.argmax(x)


# In[ ]:

def display_front():
    st.write("hello")


def main():
    display_front()



if __name__=="__main__":
    main()
