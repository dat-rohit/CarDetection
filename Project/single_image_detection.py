#imports
from ast import IsNot
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


from Model import YOLOv3
from tensorflow.keras.layers import Input
from WeightsReader import WeightReader

fig = plt.figure(figsize=(12, 12))
axis = fig.add_subplot()

#global variables

# IoU Threshold
iou_thresh = 0.6

# Objectness Threshold
obj_thresh = 0.6

# network's input dimensions
net_h, net_w = 416, 416

# 3 anchor boxes (width,height) pairs
anchors = [ [[116, 90], [156, 198], [373, 326]], 
             [[30, 61], [62, 45], [59, 119]], 
             [[10, 13], [16, 30], [33, 23]]]


NUM_CLASS = 80


global model
global graph

#graph=tf.get_default_graph()

#session = tf.Session()
#set_session(session)

# create Yolo model
@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None},allow_output_mutation=True)
def load_model():
    model = YOLOv3(Input(shape=(None, None, 3)), NUM_CLASS);    
    #model.summary()
    # load the weights trained on COCO into the model
    WeightReader("./yolov3.weights").load_weights(model)
    return model



model=load_model()
#model._make_predict_function()


#define labels
labels = ["person",        "bicycle",       "car",          "motorbike",      "aeroplane",     "bus",        "train",   "truck",           "boat",          "traffic light", "fire hydrant", "stop sign",      "parking meter", "bench",           "bird",          "cat",           "dog", "horse", "sheep",          "cow",           "elephant",   "bear",    "zebra", "giraffe",           "backpack",      "umbrella",      "handbag",      "tie",            "suitcase",      "frisbee",    "skis",    "snowboard",           "sports ball",   "kite",          "baseball bat", "baseball glove", "skateboard",    "surfboard",           "tennis racket", "bottle",        "wine glass",   "cup",            "fork",          "knife",      "spoon",   "bowl", "banana",           "apple",         "sandwich",      "orange",       "broccoli",       "carrot",        "hot dog",    "pizza",   "donut", "cake",           "chair",         "sofa",          "pottedplant",  "bed",            "diningtable",   "toilet",     "tvmonitor", "laptop", "mouse",           "remote",        "keyboard",      "cell phone",   "microwave",      "oven",          "toaster",    "sink",    "refrigerator",           "book",          "clock",         "vase",         "scissors",       "teddy bear",    "hair drier", "toothbrush"]


#Resize the image to the input's size of network

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
    
    return selected_indices, selected_scores, selected_boxes, selected_classes

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

def try_for_one_image(image):

    input = preprocess_image(image, net_h, net_w)
    #with graph.as_default():
    output = model.predict(input)
    image_h, image_w, _ = image.shape
    selected_indices, selected_scores, selected_boxes, selected_classes = ApplyModel(output)
    image, list_start, list_end, list_label = draw_boxes(image, selected_boxes,selected_classes, selected_scores, image_w, image_h, net_w, net_h)
    if len(list_label) > 1 :
            maxi = np.argmax(list_label)
            list_start[0] = list_start[maxi]
            list_end[0] = list_end[maxi]
            #fig = plt.figure(figsize=(12, 12))
            #axis = fig.add_subplot()
            case_coordonnee = np.array([list_start[0][0], list_end[0][0], list_start[0][1], list_end[0][1]])
            image = np.ascontiguousarray(image, dtype=np.uint8)
            cv2.rectangle(image, list_start[0], list_end[0], color=(0, 255, 0))    #ajouter ,3     
            cv2.putText(image, "car", (list_start[0][0], list_start[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0,0,0), 2)
 
    else :
        if (list_start[0][0]==-math.pi) and (list_start[0][1]==-math.pi):
            print('there is no car')
        else :  
            #fig = plt.figure(figsize=(12, 12))
            #axis = fig.add_subplot()
            case_coordonnee = np.array([list_start[0][0], list_end[0][0], list_start[0][1], list_end[0][1]])
            cv2.rectangle(image, list_start[0], list_end[0], color=(0, 255, 0)) #ajouter ,3         
            cv2.putText(image, "car", (list_start[0][0], list_start[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0,0,0), 2)
    
    #plt.imshow(delimit_image(image))   
    #plt.show()
    final_img=delimit_image(image)
    return final_img




#Takes the image path as an input and returns the plotted image with bounding boxes
def detect_cars(image):
    #img=Image.open(image)
    print("detect cars okkkk")
    img=image
    img=resizeimage.resize_contain(img, [400, 250])
    val=np.array(img,dtype=np.uint8)
    image=val[:,:,[0,1,2]]
    final_img = try_for_one_image(image)
    return final_img



st.header("Detect a car")

#Build the streamlit front end page
def display_front():
    file_uploaded=st.file_uploader("Choose file", type=["png","jpg","jpeg"])
    
    if file_uploaded is not None:
        #final_img=detect_cars(image=file_uploaded)
        #plt.imshow(final_img)
        #plt.axis("off")
        #st.pyplot(fig)
        #st.write("this is a car")
        image = Image.open(file_uploaded)
        print("okkkkkkkkkk")
        final_img=detect_cars(image=image)
        print("diube OKKKKKK")
        plt.imshow(final_img)
        plt.axis("off")
        st.pyplot(fig)

    st.write("ok test")
 


def main():
    display_front()


if __name__=="__main__":
    main()
