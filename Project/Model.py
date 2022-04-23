import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import add, concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import numpy as np

def darknet():
    """Darknet-53 classifier.
    """
    inputs = Input(shape=(416, 416, 3))
    x = darknet53(inputs)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='softmax')(x)

    model = Model(inputs, x)

    return model

def YOLOv3(input_layer, NUM_CLASS=80):
    """
    Yolov3 network. 
    :param input_layer:  Input layer of shape (1, 416, 416, 3)
    :param NUM_CLASS:  Should be 80 for Yolov3    
    :return:  A tensorflow.keras.models.Model object with three outputs of the Yolov3 network.
    """

    # Darknet backbone
    route_1, route_2, conv = darknet53(input_layer)

    # set layer-index 75
    idx = np.array([75])    

    conv = convolutional(conv, K=1, filters=512, layer_num=idx)
    conv = convolutional(conv, K=3, filters=1024, layer_num=idx)
    conv = convolutional(conv, K=1, filters=512, layer_num=idx)
    conv = convolutional(conv, K=3, filters=1024, layer_num=idx)
    conv = convolutional(conv, K=1, filters=512, layer_num=idx)

    conv_lobj_branch = convolutional(conv, K=3, filters=1024, layer_num=idx)
    conv_lbbox = convolutional(conv_lobj_branch, K=1, filters=(3 * (NUM_CLASS + 5)), bn=False, layer_num=idx)

    # set layer-index 84
    idx = np.array([84])    
    
    conv = convolutional(conv, K=1, filters=256, layer_num=idx)
    conv = upsample(conv)
    conv = concatenate([conv, route_2], axis=-1)

    # set layer-index 87
    idx = np.array([87])    

    conv = convolutional(conv, K=1, filters=256, layer_num=idx)
    conv = convolutional(conv, K=3, filters=512, layer_num=idx)
    conv = convolutional(conv, K=1, filters=256, layer_num=idx)
    conv = convolutional(conv, K=3, filters=512, layer_num=idx)
    conv = convolutional(conv, K=1, filters=256, layer_num=idx)

    conv_mobj_branch = convolutional(conv, K=3, filters=512, layer_num=idx)
    conv_mbbox = convolutional(conv_mobj_branch, K=1, filters=(3 * (NUM_CLASS + 5)), bn=False, layer_num=idx)

    # set layer-index 96
    idx = np.array([96])    

    conv = convolutional(conv, K=1, filters=128, layer_num=idx)
    conv = upsample(conv)
    conv = concatenate([conv, route_1], axis=-1)

    # set layer-index 99
    idx = np.array([99])    

    conv = convolutional(conv, K=1, filters=128, layer_num=idx)
    conv = convolutional(conv, K=3, filters=256, layer_num=idx)
    conv = convolutional(conv, K=1, filters=128, layer_num=idx)
    conv = convolutional(conv, K=3, filters=256, layer_num=idx)
    conv = convolutional(conv, K=1, filters=128, layer_num=idx)

    conv_sobj_branch = convolutional(conv, K=3, filters=256, layer_num=idx)
    conv_sbbox = convolutional(conv_sobj_branch, K=1, filters=(3 * (NUM_CLASS + 5)), bn=False, layer_num=idx)

    return Model( input_layer, [conv_lbbox, conv_mbbox, conv_sbbox])

def darknet53(x):
    """
    Darknet-53 Backbone.
    """

    idx = np.array([0]) # layer-index = Zero

    x = convolutional(x, K=3, filters=32, layer_num=idx)
    x = convolutional(x, K=3, filters=64, downsample=True, layer_num=idx)
    x = residual_block(x,  32, 64, layer_num=idx, repeat=1 )

    x = convolutional(x, K=3, filters=128, downsample=True, layer_num=idx)
    x = residual_block(x, 64, 128, layer_num=idx, repeat=2)

    x = convolutional(x, K=3, filters=256, downsample=True, layer_num=idx)
    x = residual_block(x, 128, 256, layer_num=idx, repeat=8)    
    route_1 = x

    x = convolutional(x, K=3, filters=512, downsample=True, layer_num=idx)
    x = residual_block(x, 256, 512, layer_num=idx, repeat=8)
    route_2 = x

    x = convolutional(x, K=3, filters=1024, downsample=True, layer_num=idx)
    x = residual_block(x, 512, 1024, layer_num=idx, repeat=4)

    return route_1, route_2, x


def convolutional(x, K, filters, downsample=False, bn=True, layer_num=[0]):
    """
    convolutional Represents a Convolutional layer of Yolov3. This is a Conv2D followed by BatchNorm and LeakyRelu 
    :param x: Input layer to be used for this convolutional layer.  
    :param K: Filter shape (K,K) 
    :param filters:  Total no. of filters to be used.
    :param downsample:  A boolean representing weather to downsample the output feature-map by 2 or not.
    :param bn: A boolean representing weather to use BatachNormalizarion layer or not. 
    :param layer_num:  The layer number for this layer. This value will be incremented by 1 at each call to this method.
    :return: 
    
    """

    input_layer = x
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    layer_name_conv  = 'conv_'  + str(layer_num[0])
    layer_name_batch = 'bnorm_' + str(layer_num[0])
    layer_name_relu  = 'leaky_' + str(layer_num[0])

    conv = tf.keras.layers.Conv2D(  kernel_size=K,
                                    filters=filters,
                                    strides=strides,
                                    padding=padding,
                                    use_bias=not bn,
                                    #kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                    #kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                    #bias_initializer=tf.constant_initializer(0.),
                                    name=layer_name_conv
                                )(input_layer)

    if bn:
        conv = tf.keras.layers.BatchNormalization(epsilon=0.001, name=layer_name_batch)(conv)
        conv = LeakyReLU(alpha=0.1, name=layer_name_relu)(conv)

    layer_num[0] += 1

    return conv

def residual_block(x, filter_num1, filter_num2, layer_num=[0], repeat=1):
    for i in range(repeat):
        short_cut = x
        conv = convolutional(x,    K=1, filters=filter_num1, layer_num=layer_num)
        conv = convolutional(conv, K=3, filters=filter_num2, layer_num=layer_num)
        residual_output = add([short_cut , conv])
        layer_num[0] += 1
        x = residual_output
    return residual_output

def upsample(x):
    return UpSampling2D(2)(x)

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def extract_features( output, anchors, S, N=3, num_classes=(80), net_wh=(416,416)):
    """
    ExtractFeatures takes the raw network output and extracts the bouding-box coordinates, class probabilites, and
    Objectness scores information, using the provided grid-size and anchor-boxes information.   
    :param output: raw network output from Yolo for a particular grid-scale.
    :param anchors: the anchor box dimensions for this grid-scale
    :param S: the grid-scale. This should be 13, 26 or 52.
    :param N: the number of anchor boxes per grid cell.
    :param num_classes: The total number of classes. This should be 80 for Yolov3.
    :param net_wh: a Tuple containing network input width height. This should be (416, 416)
    :return:  2D and 1D Tensors of shape [num_boxes, :] for boxes, scores and class-probabilities for all boxes in this grid scale. 
    """    

    # netout: reshape (X,X,255) to (X,X,3,85) 
    netout = output.reshape((S, S, N, -1))

    # reshape anchors into a tensor of shape (1, 1, 3, 2) so it can be multiplied with later
    anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)    
    t_wh = netout[..., 2:4]                   # tw,th = (X,X,3, [2,3]).
    box_wh = np.exp(t_wh) * anchors_tensor    # Then Bw = e^tw * anchor_w,  a tensor of shape: (X,X,3,2) * (1,1,3,2)
    box_wh /= net_wh

    # create an S*S grid of rows, cols indices of shape (X,X,3,1)
    col = np.tile(np.arange(0, S), S).reshape(-1, S)
    col = col.reshape(S, S, 1, 1).repeat(3, axis=-2)
    row = np.tile(np.arange(0, S).reshape(-1, 1), S)
    row = row.reshape(S, S, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)                        

    # sigmoid netout tx,ty = (X,X,3, [0,1]). Then Bx,By = sigmoid(tx), sigmoid(ty) 
    box_xy = _sigmoid(netout[..., :2])
    box_xy += grid            # Bx = (col + x)
    box_xy /= (S, S)          # Bx = Bx / S
    box_xy -= (box_wh / 2.)
    boxes = np.concatenate((box_xy, box_wh), axis=-1)    # final tensor shape (S, S, 3, 4)
        
    box_confidence = _sigmoid(netout[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = _sigmoid(netout[..., 5:])

    boxes = boxes.reshape((-1,4))                        # reshape to A 2-D float Tensor of shape [num_boxes, 4].
    scores = box_confidence.reshape(-1)                  # reshape to A 1-D float Tensor of shape [num_boxes].
    classes = box_class_probs.reshape(-1, num_classes)   # reshape to A 2-D float Tensor of shape [num_boxes, num_classes].
    
    return boxes, scores, classes