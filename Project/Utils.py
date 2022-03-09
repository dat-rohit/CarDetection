import numpy as np
import cv2
import random
from random import randrange

labels = ["person",        "bicycle",       "car",          "motorbike",      "aeroplane",     "bus",        "train",   "truck", \
          "boat",          "traffic light", "fire hydrant", "stop sign",      "parking meter", "bench", \
          "bird",          "cat",           "dog", "horse", "sheep",          "cow",           "elephant",   "bear",    "zebra", "giraffe", \
          "backpack",      "umbrella",      "handbag",      "tie",            "suitcase",      "frisbee",    "skis",    "snowboard", \
          "sports ball",   "kite",          "baseball bat", "baseball glove", "skateboard",    "surfboard", \
          "tennis racket", "bottle",        "wine glass",   "cup",            "fork",          "knife",      "spoon",   "bowl", "banana", \
          "apple",         "sandwich",      "orange",       "broccoli",       "carrot",        "hot dog",    "pizza",   "donut", "cake", \
          "chair",         "sofa",          "pottedplant",  "bed",            "diningtable",   "toilet",     "tvmonitor", "laptop", "mouse", \
          "remote",        "keyboard",      "cell phone",   "microwave",      "oven",          "toaster",    "sink",    "refrigerator", \
          "book",          "clock",         "vase",         "scissors",       "teddy bear",    "hair drier", "toothbrush"]

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
    
    for i in range(len(boxes)):        
        r = randrange(255)
        g = randrange(255)
        b = randrange(255)                            
        color = (r,g,b)        
        label = np.argmax(classes[i,:], axis=0)

        cv2.rectangle(image, start[i], end[i], color, 3)            
        cv2.putText(image, labels[label], (start[i][0], start[i][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (0,0,0), 2)
    return image      