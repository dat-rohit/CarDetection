#%%


#You have to be able to import all of theses libraries
import math
import numpy as np
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt
import cv2
from Utils import preprocess_image
from Model import extract_features
import tensorflow as tf
from Model import YOLOv3
from tensorflow.keras.layers import Input
from WeightsReader import WeightReader

ImageFile.LOAD_TRUNCATED_IMAGES = True
# create Yolo model
model = YOLOv3(Input(shape=(None, None, 3)));
#model.summary()

# load the weights trained on COCO into the model
WeightReader("./yolov3.weights").load_weights(model)

iou_thresh = 0.6

# Objectness Threshold 0.6
obj_thresh = 0.6

# network's input dimensions
net_h, net_w = 416, 416

# 3 anchor boxes (width,height) pairs
anchors = [ [[116, 90], [156, 198], [373, 326]],
             [[30, 61], [62, 45], [59, 119]],
             [[10, 13], [16, 30], [33, 23]]]



def preprocess_image(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h

        # resize the image to the new size. Normalize the data and reflect the pixels [:,:,::-1]
    resized = cv2.resize(image[:, :, ::-1] / 255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2),
    int((net_w - new_w) // 2):int((net_w + new_w) // 2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def getScaleFactors(image_w, image_h, net_w, net_h):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_h = (image_h * net_w) / image_w
        new_w = net_w
    else:
        new_w = (image_w * net_h) / image_h
        new_h = net_h

    x_scale = float(new_w) / net_w
    y_scale = float(new_h) / net_h
    x_offset = (net_w - new_w) / 2. / net_w
    y_offset = (net_h - new_h) / 2. / net_h

    return x_scale, y_scale, x_offset, y_offset


def draw_boxes(image, boxes, classes, scores, image_w, image_h, net_w, net_h):
    x = boxes[:, 0].numpy()
    y = boxes[:, 1].numpy()
    w = boxes[:, 2].numpy()
    h = boxes[:, 3].numpy()

    x2 = x + w
    y2 = y + h

    x_scale, y_scale, x_offset, y_offset = getScaleFactors(image_w, image_h, net_w, net_h)

    x = (x - x_offset) / x_scale * image_w
    x2 = (x2 - x_offset) / x_scale * image_w
    y = (y - y_offset) / y_scale * image_h
    y2 = (y2 - y_offset) / y_scale * image_h

    start = list(zip(x.astype(int), y.astype(int)))
    end = list(zip(x2.astype(int), y2.astype(int)))
    list_start = []
    list_end = []
    list_label = []
    for i in range(len(boxes)):
        label = np.argmax(classes[i, :], axis=0)
        if (labels[label] == "car" or labels[label] == "truck" or labels[label] == "bus"):
            # cv2.rectangle(image, start[i], end[i], color, 3)
            proba_label = np.max(classes[i, :], axis=0)
            list_start.append(start[i])
            list_end.append(end[i])
            list_label.append(proba_label)
            # cv2.putText(image, labels"car", (start[i][0], start[i][1] - 10),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            # (0,0,0), 2)
    if len(list_start) == 0:
        list_start.append((-math.pi, -math.pi))
        list_end.append((-math.pi, -math.pi))
        list_label.append((-math.pi, -math.pi))
    return image, list_start, list_end, list_label


def ApplyModel(output):
    boxes = np.empty([1, 4])
    scores = np.empty([1, ])
    classes = np.empty([1, 80])

    for i in range(len(output)):
        _, S = output[i].shape[:2]

        b, s, c = extract_features(output[i], anchors[i], S, N=3, num_classes=(80), net_wh=(416, 416))

        boxes = np.concatenate((boxes, b), axis=0)
        scores = np.concatenate((scores, s), axis=0)
        classes = np.concatenate((classes, c), axis=0)

    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes, scores, len(boxes),
        iou_threshold=iou_thresh,
        score_threshold=obj_thresh,
        soft_nms_sigma=0.6)

    selected_boxes = tf.gather(boxes, selected_indices)
    selected_classes = tf.gather(classes, selected_indices)

    # A adapter pour pouvoir avoir qu'une fonction excécutant notre model
    # '''  image', list_start, list_end, list_label '= draw_boxes(X[k], selected_boxes,selected_classes, selected_scores, image_w, image_h, net_w, net_h)
    # cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8'))
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


def create_dataset_for_second_model(X, y):
    X_new = []
    y_valid = []
    y_test = []
    car_model_new = []
    for k in range(len(X)):
        input = preprocess_image(X[k], net_h, net_w)
        output = model.predict(input)
        selected_indices, selected_scores, selected_boxes, selected_classes = ApplyModel(output)
        image, list_start, list_end, list_label = draw_boxes(X[k], selected_boxes, selected_classes, selected_scores,
                                                             image_w, image_h, net_w, net_h)
        # cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8'))
        if (list_start[0][0] == -math.pi) and (list_start[0][1] == -math.pi):
            '''print('there is no car')'''
        else:
            maxi = np.argmax(list_label)
            list_start[0] = list_start[maxi]
            list_end[0] = list_end[maxi]
            # fig = plt.figure(figsize=(12, 12))
            # axis = fig.add_subplot()
            case_coordonnee = np.array([list_start[0][0], list_end[0][0], list_start[0][1], list_end[0][1]])
            # cv2.rectangle(image, list_start[0], list_end[0], color=(0, 255, 0))    #ajouter ,3
            # cv2.putText(image, "car", (list_start[0][0], list_start[0][1] - 10),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.5, #(0,0,0), 2)
            X_new.append(X[k])
            y_test.append(case_coordonnee)
            y_valid.append(y[k])

    X_new = np.array(X_new)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)
    car_model_new = np.array(car_model_new)

    return X_new, y_test, y_valid, car_model_new


def try_for_one_image(image, image_h, image_w):
    input = preprocess_image(image, net_h, net_w)
    output = model.predict(input)
    #image_h, image_w, _ = a.shape
    selected_indices, selected_scores, selected_boxes, selected_classes = ApplyModel(output)
    image, list_start, list_end, list_label = draw_boxes(image, selected_boxes, selected_classes, selected_scores, image_w,
                                                         image_h, net_w, net_h)

    if len(list_label) > 1:
        maxi = np.argmax(list_label)
        list_start[0] = list_start[maxi]
        list_end[0] = list_end[maxi]
        fig = plt.figure(figsize=(12, 12))
        axis = fig.add_subplot()
        case_coordonnee = np.array([list_start[0][0], list_end[0][0], list_start[0][1], list_end[0][1]])
        image = np.ascontiguousarray(image, dtype=np.uint8)
        cv2.rectangle(image, list_start[0], list_end[0], color=(0, 255, 0))  # ajouter ,3
        cv2.putText(image, "car", (list_start[0][0], list_start[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2)

    else:
        if (list_start[0][0] == -math.pi) and (list_start[0][1] == -math.pi):
            print('there is no car')
        else:
            fig = plt.figure(figsize=(12, 12))
            axis = fig.add_subplot()
            case_coordonnee = np.array([list_start[0][0], list_end[0][0], list_start[0][1], list_end[0][1]])
            image = np.ascontiguousarray(image, dtype=np.uint8)
            cv2.rectangle(image, list_start[0], list_end[0], color=(0, 255, 0))  # ajouter ,3
            cv2.putText(image, "car", (list_start[0][0], list_start[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2)
    plt.imshow(delimit_image(image))
    plt.show()
    return


def delimit_image(image):
    color_1 = np.array(image)
    color_1 = color_1.mean(axis=2)
    mask = color_1 < 255

    pos2 = 0
    for i in range(len(mask)):
        for j in range(len(mask[1])):
            if (mask[i, j] == True):
                pos2 = (i, j)
    pos1 = 0
    for i in range(len(mask)):
        for j in range(len(mask[1])):
            if (mask[i, j] == True) and (pos1 == 0):
                pos1 = (i, j)

    return image[pos1[0]:pos2[0] + 1, pos1[1]:pos2[1] + 1, :]


def delimit_box(image_delimited):
    color_0 = np.array(image[:, :, 0])
    mask0 = color_0 == 0

    color_1 = np.array(image[:, :, 1])
    mask1 = color_1 == 255

    color_2 = np.array(image[:, :, 2])
    mask2 = color_2 == 0

    pos2 = 0
    for i in range(1, len(mask1)):
        for j in range(1, len(mask1[1])):
            if (mask1[i, j] == True) and (mask0[i, j] == True) and (mask2[i, j] == True):
                pos2 = (i, j)
    pos1 = 0
    for i in range(1, len(mask1)):
        for j in range(1, len(mask1[1])):
            if (mask1[i, j] == True) and (mask0[i, j] == True) and (mask2[i, j] == True) and (pos1 == 0):
                pos1 = (i, j)

    return pos1[0], pos2[0], pos1[1], pos2[1]


def convert_ratio(image_origine_shape, image):
    image = delimit_image(image)
    xmin, xmax, ymin, ymax = delimit_box(image)
    ymin_f = ymin * image_origine_shape[0] / image.shape[0]
    ymax_f = ymax * image_origine_shape[0] / image.shape[0]
    xmin_f = xmin * image_origine_shape[1] / image.shape[1]
    xmax_f = xmax * image_origine_shape[1] / image.shape[1]
    return [xmin_f, xmax_f, ymin_f, ymax_f]

def resize_contain(image, size, resample=Image.LANCZOS, bg_color=(255, 255, 255, 0)):
    """
    Resize image according to size.
    image:      a Pillow image instance
    size:       a list of two integers [width, height]
    """
    img_format = image.format
    img = image.copy()
    img.thumbnail((size[0], size[1]), resample)
    background = Image.new('RGBA', (size[0], size[1]), bg_color)
    img_position = (
        int(math.ceil((size[0] - img.size[0]) / 2)),
        int(math.ceil((size[1] - img.size[1]) / 2))
    )
    background.paste(img, img_position)
    background.format = img_format
    return background.convert('RGBA')

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

def detect_car(image_file = 'voiture.jpg'):
    image = Image.open(image_file)
    image = resize_contain(image, [400, 250])
    image = np.array(image, dtype=np.uint8)
    image = image[:, :, [0, 1, 2]]
    image_h, image_w, _ = image.shape
    try_for_one_image(image,image_h, image_w)
    return

detect_car()
