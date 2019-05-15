import os, sys, json
import argparse
import math
import pprint as pp
import statistics
import copy
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import Session

# yolo.frontend 
from keras import backend as K
from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, Lambda, Activation, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transf



class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None, label=-1, score=-1):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = label
        self.score = score

        self.filtered = False

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

    def get_json(self):
        return ([self.xmin, self.xmax, self.ymin, self.ymax, ind_to_label[int(self.label)], str(int(100*self.score))])

    def __str__(self): 
        return str(self.xmin,) + '   ' + str(self.xmax) + '   ' + str(self.ymin) + '   ' + str(self.ymax) + '   ' + str(self.c) + '   ' + str(self.label) + '   ' + str(self.classes) + '   ' + str(str(self.score))


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


#yolo utils
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union

def bbox_iou_right_join(box1, box2):
    """ Returns right-join ratio IOU (not original IOU)
    Usually box1 would be bucket and box2 can be either Wear Area or MatInside
    """
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    left_join = w2 * h2
    left_join_ratio = float(intersect) / left_join
    return left_join_ratio

def draw_boxes(image, boxes, labels, score_threshold=0.50,
    class_obj_threshold=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]):
    image_h, image_w, _ = image.shape
    colors = [(0, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (0, 255, 255)]

    for box in boxes:
        # if box.get_score() < class_obj_threshold[box.get_label()]: continue
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)
        thickness = 1 if box.filtered else 2
        if box.filtered: continue
        thickness = 1

        # if xmin < -15 or xmax < 0 or ymin < -15 or ymax < 0:
        #     # print(xmin, xmax, ymin, ymax)
        #     continue
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        color = colors[box.get_label()]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color,
                      thickness)
        font_increase = 2.5 if box.get_label() == 3 else 1.
        cv2.putText(image,
                    # labels[box.get_label()] + ' ' + str(box.get_score())[:4],
                    str(box.get_score())[:4],
                    (xmin, ymin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    8e-4 * image_h * font_increase,
                    color, thickness)

    return image

def decode_netout(netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.01,class_obj_threshold=None):
    if class_obj_threshold is None:
        class_obj_threshold = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    grid_h, grid_w, nb_box = netout.shape[:3]
    boxes = []

    # decode the output by the network
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + _sigmoid(x)) / grid_w  # center position, unit: image width
                    y = (row + _sigmoid(
                        y)) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height
                    confidence = netout[row, col, b, 4]

                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence,
                                   classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if
             box.get_score() > class_obj_threshold[box.get_label()]]

    return boxes

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x / np.min(x) * t
    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)

def convert_bbox_coords_to_pixels(bbox, image_w=640, image_h=640):
    xmin = bbox.xmin * image_w
    xmax = bbox.xmax * image_w
    ymin = bbox.ymin * image_h
    ymax = bbox.ymax * image_h
    return xmin, xmax, ymin, ymax

def tf_predict(sess, y, x, image, image_path, pb_filepath):
    obj_threshold = 0.3
    nms_threshold = 0.01
    image_h, image_w, _ = image.shape
    image = cv2.resize(image, (640, 640))
    image = image / 255.
    # image = self.feature_extractor.normalize(image)

    input_image = image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)
    input_image = np.expand_dims(input_image, 0)

    # netout = self.model.predict(input_image)[0]
    start_time = time()
    netout = sess.run(y, {x: input_image})[0]

    anchors = [1.27,2.0,  0.83,1.24,  0.48,0.54,  0.60,1.3, 7.30,1.60,  7.23,3.13,  8.96,4.44,   13.47,9.82,  11.32,7.42]
    
    labels = ["tooth", "toothline"]
    nb_class = len(labels)
    boxes = decode_netout(netout, anchors, nb_class, obj_threshold, nms_threshold)
    print("Time spent: ", time() - start_time)

    image = draw_boxes(image, boxes, labels, obj_threshold)
    # plt.imshow(image)
    # plt.show()
    path_to_save = pb_filepath.split('/')[:-1]
    path_to_save = '/'.join(path_to_save)
    path_to_save = os.path.join(path_to_save, image_path.split('/')[-1])
    print(path_to_save, image.shape)
    image *= 255
    cv2.imwrite(path_to_save, image)

    return boxes

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.device('/device:GPU:0'):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
    return graph


#backend
class FullYoloFeature(object):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
        def space_to_depth_x2(x):
            import tensorflow as tf
            return tf.space_to_depth(x, block_size=2)

        # Layer 1
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1',
                   use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same',
                                 name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)
        # self.feature_extractor.load_weights(FULL_YOLO_BACKEND_PATH)

        # self.feature_extractor.summary()
        print('\n')

    def normalize(self, image):
        return image / 255.

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)

#frontend 
class YOLO(object):
    def __init__(
        self,
        input_size,
        labels,
        max_box_per_image,
        anchors):

        self.input_size = input_size

        self.labels = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box = len(anchors) // 2
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors = anchors

        self.max_box_per_image = max_box_per_image

        # make the feature extractor layers
        input_image = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))


        self.feature_extractor = FullYoloFeature(self.input_size)
  

        print("Output Shape of feature extractor: ",
              self.feature_extractor.get_output_shape())
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()
        features = self.feature_extractor.extract(input_image)

        # make the object detection layer
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self.grid_h, self.grid_w, self.nb_box,
                          4 + 1 + self.nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output)

        # initialize the weights of the detection layer
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape) / (self.grid_h * self.grid_w)
        new_bias = np.random.normal(size=weights[1].shape) / (self.grid_h * self.grid_w)

        layer.set_weights([new_kernel, new_bias])

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def predict(
        self,
        image,
        obj_threshold,
        nms_threshold,
        is_filter_bboxes,
        shovel_type,
        class_obj_threshold):

        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:, :, ::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1, 1, 1, 1, self.max_box_per_image, 4))

        netout = self.model.predict([input_image, dummy_array])[0]

        boxes = decode_netout(netout, self.anchors, self.nb_class,
                              obj_threshold=obj_threshold,
                              nms_threshold=nms_threshold,
                              class_obj_threshold=class_obj_threshold)

        filtered_boxes = []
        if is_filter_bboxes:
            boxes, filtered_boxes = filter_all_objects(boxes, shovel_type,
                                                       image_size=image_h)

        return boxes, filtered_boxes

    def get_inference_model(self):
        inference_model = Model(self.model.input,
                                self.model.get_layer("reshape_1").output)
        return inference_model

    def get_feature_model(self, is_before_activation=True):
        if is_before_activation:
            feature_model = Model(self.feature_extractor.feature_extractor.inputs,
                                  self.feature_extractor.feature_extractor. \
                                  get_layer("conv_22").output)
        else:
            feature_model = Model(self.feature_extractor.feature_extractor.inputs,
                                  self.feature_extractor.feature_extractor. \
                                  get_layer("leaky_re_lu_22").output)
        return feature_model

def normalize(image):
    return image / 255.

# post_process_detections
thresholds = \
    {
        "Hydraulic":
            {
                "num_pixel_tooth_threshold": 11,
                "min_area_threshold":
                    {
                        "Tooth": 13 * 13,
                        "Toothline": 200 * 40,
                        "BucketBB": 120 * 120,
                        "MatInside": 200 * 100,
                        "WearArea": 150 * 100,
                        "LipShroud": 18 * 18
                    },
                "max_area_threshold":
                    {
                        "Tooth": 75 * 75,
                        "Toothline": 500 * 250,
                        "BucketBB": 640 * 640,
                        "MatInside": 500 * 500,
                        "WearArea": 400 * 200,
                        "LipShroud": 70 * 50
                    }
            },
        "Cable": {
            "num_pixel_tooth_threshold": 6,
            "min_area_threshold":
                {
                    "Tooth": 12 * 12,
                    "Toothline": 200 * 40,
                    "BucketBB": 110 * 110,
                    "MatInside": 200 * 100,
                    "WearArea": 100 * 100,
                    "LipShroud": 18 * 18
                },
            "max_area_threshold":
                {
                    "Tooth": 50 * 50,
                    "Toothline": 500 * 250,
                    "BucketBB": 640 * 640,
                    "MatInside": 500 * 500,
                    "WearArea": 400 * 300,
                    "LipShroud": 70 * 50
                }
        },
        "Backhoe": {
            "num_pixel_tooth_threshold": 8,
            "min_area_threshold":
                {
                    "Tooth": 12 * 12,
                    "Toothline": 200 * 40,
                    "BucketBB": 150 * 120,
                    "MatInside": 200 * 100,
                    "WearArea": 100 * 100,
                    "LipShroud": 18 * 18
                },
            "max_area_threshold":
                {
                    "Tooth": 50 * 50,
                    "Toothline": 500 * 250,
                    "BucketBB": 640 * 640,
                    "MatInside": 500 * 500,
                    "WearArea": 400 * 300,
                    "LipShroud": 70 * 50
                }
        }
    }
# aspect_ratio = width / height
min_aspect_ratios = {
    "Tooth": 0.35,
    "Toothline": 1.25,
    "BucketBB": 0.7,
    "MatInside": 0.5,
    "WearArea": 1.25,
    "LipShroud": 0.35}
max_aspect_ratios = {
    "Tooth": 1.4,
    "Toothline": 6.0,
    "BucketBB": 5.0,
    "MatInside": 3.0,
    "WearArea": 4.0,
    "LipShroud": 1.3}
ind_to_label = {
    0: "Tooth",
    1: "Toothline",
    2: "BucketBB",
    3: "MatInside",
    4: "WearArea",
    5: "LipShroud"}
ind_to_label = {
    0: "Tooth",
    1: "Toothline",
    2: "BucketBB",
    3: "MatInside",
    4: "WearArea",
    5: "LipShroud"
}

def is_object_present(bboxes, class_ind):
    is_present = False
    for bbox in bboxes:
        if bbox.get_label() == class_ind:
            is_present = True

    return is_present

def get_best_bbox_of_class(bboxes, class_index):
    """ Looks for a single highest-probability instance of class class_index """
    obj_to_find_score, obj_to_find_bbox = 0., None
    new_bboxes = []
    filtered_bboxes = []
    for bbox in bboxes:
        if bbox.get_label() == class_index and bbox.get_score() > obj_to_find_score:
            obj_to_find_score = bbox.get_score()
            obj_to_find_bbox = bbox
        elif bbox.get_label() == class_index and bbox.get_score() <= obj_to_find_score:
            filtered_bboxes.append(bbox)
        else:
            new_bboxes.append(bbox)
    if obj_to_find_bbox:
        new_bboxes.append(obj_to_find_bbox)

    return new_bboxes, obj_to_find_bbox, filtered_bboxes

def filter_wrong_sizes(bboxes, image_size, min_area_thresholds, max_area_thresholds,
    min_aspect_ratios, max_aspect_ratios):
    """ Removes bboxes of wrong area or aspect ratio. """
    image_w, image_h = image_size, image_size

    filtered_bboxes = []
    for bbox in bboxes:
        class_ind = bbox.get_label()
        class_string = ind_to_label[class_ind]
        min_area_threshold = min_area_thresholds[class_string]
        max_area_threshold = max_area_thresholds[class_string]
        min_aspect_ratio = min_aspect_ratios[class_string]
        max_aspect_ratio = max_aspect_ratios[class_string]

        xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(bbox, image_w, image_h)
        width = xmax - xmin
        height = ymax - ymin
        bbox_area = width * height
        bbox_aspect = float(width) / height
        if bbox_area < min_area_threshold or bbox_area > max_area_threshold or \
                bbox_aspect < min_aspect_ratio or bbox_aspect > max_aspect_ratio:
            bboxes.remove(bbox)
            filtered_bboxes.append(bbox)

    return bboxes, filtered_bboxes

def filter_teeth_outside_toothline(bboxes, teethline_bbox, x_threshold=40.,
    y_threshold=20.):
    # refine tooth bboxes with teethline because teethline dets are more accurate
    filtered_bboxes = []
    if teethline_bbox:
        teethline_bbox_xmin, teethline_bbox_xmax, teethline_bbox_ymin, \
        teethline_bbox_ymax = convert_bbox_coords_to_pixels(teethline_bbox)
        bboxes_to_return = []
        for bbox in bboxes:
            xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(bbox)
            if bbox.get_label() == 0:
                if xmin < teethline_bbox_xmin - x_threshold or \
                        xmax > teethline_bbox_xmax + x_threshold or \
                        ymin < teethline_bbox_ymin - y_threshold or \
                        ymax > teethline_bbox_ymax + y_threshold:
                    filtered_bboxes.append(bbox)
                else:
                    bboxes_to_return.append(bbox)
            else:
                bboxes_to_return.append(bbox)
    else:
        bboxes_to_return = []
        filtered_bboxes = bboxes
    return bboxes_to_return, filtered_bboxes

def filter_inside_bucket(bucket_bbox, bbox_to_check, right_join_thresh=0.7):
    if bbox_iou_right_join(bucket_bbox, bbox_to_check) < right_join_thresh:
        return None
    else:
        return bbox_to_check

def filter_close_teeth(bboxes, pixel_threshold=4):
    """ Get rids of FPs that are sometimes predicted between the teeth along the
    toothline. """
    # sort bboxes by x-coordinate
    other_bboxes = [bbox for bbox in bboxes if bbox.get_label() != 0]
    teeth_bboxes = [bbox for bbox in bboxes if bbox.get_label() == 0]
    teeth_xmins = [bbox.xmin for bbox in teeth_bboxes]
    xmin_sorted = np.argsort(teeth_xmins)
    teeth_bboxes = np.array(teeth_bboxes)[xmin_sorted]
    teeth_bboxes = teeth_bboxes.tolist()
    filtered_bboxes = []

    # get rid of bboxes which are within pixel_threshold in the x-axis
    i = 0
    if teeth_bboxes:
        teeth_bboxes_to_return = [teeth_bboxes[0]]
    else:
        teeth_bboxes_to_return = []
    if len(teeth_bboxes) > 2:
        while i + 2 < len(teeth_bboxes):  # iterate over the triples
            bbox1 = teeth_bboxes[i]
            bbox2 = teeth_bboxes[i + 1]
            bbox3 = teeth_bboxes[i + 2]
            xmin1, xmax1, ymin1, ymax1 = convert_bbox_coords_to_pixels(bbox1)
            xmin2, xmax2, ymin2, ymax2 = convert_bbox_coords_to_pixels(bbox2)
            xmin3, xmax3, ymin3, ymax3 = convert_bbox_coords_to_pixels(bbox3)
            if xmin2 - xmax1 < pixel_threshold and xmin3 - xmax2 < pixel_threshold:
                filtered_bboxes.append(bbox2)
            else:
                teeth_bboxes_to_return.append(teeth_bboxes[i + 1])
            i += 1

    if len(teeth_bboxes) > 1:
        teeth_bboxes_to_return.append(teeth_bboxes[-1])

    bboxes_to_return = teeth_bboxes_to_return + other_bboxes
    return bboxes_to_return, filtered_bboxes

def filter_enough_teeth(bboxes, num_teeth):
    # only predict teeth when enough teeth bboxes are observed 
    num_teeth_threshold = num_teeth - 2
    num_of_teeth_dets = 0
    for bbox in bboxes:
        if bbox.get_label() == 0:
            num_of_teeth_dets += 1
    if num_of_teeth_dets <= num_teeth_threshold:
        bboxes = [bbox for bbox in bboxes if bbox.get_label() != 0]

    return bboxes

def filter_low_scoring_teeth(bboxes, teeth_in_bucket=9):
    """ Remove lowest scoring teeth if the detected number of teeth is larger than there
    should be. """
    other_bboxes = [bbox for bbox in bboxes if bbox.get_label() != 0]
    teeth_bboxes = [bbox for bbox in bboxes if bbox.get_label() == 0]
    teeth_scores = [bbox.get_score() for bbox in teeth_bboxes]
    scores_sorted = np.argsort(teeth_scores)
    sorted_teeth_bboxes = np.array(teeth_bboxes)[scores_sorted]
    top_teeth_bboxes = sorted_teeth_bboxes[:teeth_in_bucket].tolist()
    filtered_teeth_bboxes = sorted_teeth_bboxes[teeth_in_bucket:].tolist()

    bboxes = top_teeth_bboxes + other_bboxes
    return bboxes, filtered_teeth_bboxes

def filter_all_objects(bboxes, shovel_type, num_teeth, image_size=640):
    """ Excludes many possible cases of False Positives. """
    image_w, image_h = image_size, image_size
    shovel_thresholds = thresholds[shovel_type]

    all_filtered_bboxes = []
    bboxes, size_filtered_bboxes = filter_wrong_sizes(bboxes, image_size,
                                                      shovel_thresholds[
                                                          "min_area_threshold"],
                                                      shovel_thresholds[
                                                          "max_area_threshold"],
                                                      min_aspect_ratios,
                                                      max_aspect_ratios)
    # seperate the teeth and toothline bounding boxes from bucket, matInside, and wearArea boxes
    teeth_toothline_bboxes = [bbox for bbox in bboxes if bbox.get_label() == 0 or \
                              bbox.get_label() == 1]
    other_bboxes = [bbox for bbox in bboxes if bbox.get_label() != 0 and \
                    bbox.get_label() != 1]
    teeth_toothline_bboxes, tooth_filtered_bboxes = filter_teeth_toothline(
        teeth_toothline_bboxes,
        shovel_thresholds, num_teeth)
    other_bboxes, filtered_other_bboxes = filter_bucket_mat_inside_wear_area(other_bboxes)
    all_filtered_bboxes += size_filtered_bboxes + tooth_filtered_bboxes + \
                           filtered_other_bboxes
    for bbox in all_filtered_bboxes:
        if not bbox:
            all_filtered_bboxes.remove(bbox)
        else:
            bbox.filtered = True

    good_bboxes = teeth_toothline_bboxes + other_bboxes
    return good_bboxes, all_filtered_bboxes

def filter_teeth_toothline(bboxes, shovel_thresholds, num_teeth):
    toothline_class_ind = 1
    new_bboxes, teethline_bbox, filtered_toothlines = get_best_bbox_of_class(
        bboxes, toothline_class_ind)
    new_bboxes, filtered_teeth = filter_teeth_outside_toothline(new_bboxes,
                                                                teethline_bbox)
    # new_bboxes = filter_enough_teeth(new_bboxes, num_teeth)
    new_bboxes, filtered_close_teeth = filter_close_teeth(new_bboxes,
                                                          pixel_threshold=
                                                          shovel_thresholds[
                                                              "num_pixel_tooth_threshold"])
    new_bboxes, filtered_redundant_teeth = filter_low_scoring_teeth(new_bboxes,
                                                                    teeth_in_bucket=num_teeth)
    filtered_teeth_toothline = filtered_toothlines + filtered_teeth + \
                               filtered_close_teeth + filtered_redundant_teeth
    return new_bboxes, filtered_teeth_toothline

def filter_bucket_mat_inside_wear_area(bboxes):
    # find the top single bbox of each Bucket/MatInside/WearArea class
    new_bboxes, bucket_bbox, filtered_bucket_bboxes = get_best_bbox_of_class(
        bboxes, 2)
    new_bboxes, matinside_bbox, filtered_matinside_bboxes = get_best_bbox_of_class(
        new_bboxes, 3)
    new_bboxes, weararea_bbox, filtered_weararea_bboxes = get_best_bbox_of_class(
        new_bboxes, 4)

    filtered_bboxes = []
    if bucket_bbox:
        bboxes_to_return = [bucket_bbox]
    else:
        bboxes_to_return = []

    # make sure matinside and wear area are inside bucket bounding box and only 
    # one of matinside or weararea is detected
    if bucket_bbox and weararea_bbox:
        weararea_bbox = filter_inside_bucket(bucket_bbox, weararea_bbox,
                                             right_join_thresh=0.7)
        if weararea_bbox:
            bboxes_to_return += [weararea_bbox]
            # filtered_bboxes += [matinside_bbox]
    if bucket_bbox and matinside_bbox:  # and not weararea_bbox:
        matinside_bbox = filter_inside_bucket(bucket_bbox, matinside_bbox,
                                              right_join_thresh=0.33)
        if matinside_bbox:
            bboxes_to_return += [matinside_bbox]
        else:
            filtered_bboxes += [matinside_bbox]

    filtered_bboxes += filtered_bucket_bboxes + filtered_matinside_bboxes + \
                       filtered_weararea_bboxes
    return bboxes_to_return, filtered_bboxes



# keypoint detection utils
def get_patch_margins_intertooth(boxes, image_w=640, image_h=640, x_factor=1.3, y_bottom_factor=3.):
    """
    Selects margins in pixels to select for detections. Uses intertooth distances.
    :param boxes: list of boxes
    :param image_w:
    :param image_h:
    :param x_factor: multiply the x_margin by x_factor
    :param y_bottom_factor: multiply the y_bottom margin by y_bottom_factor
    :return: 3 integers in pixel distances
    """
    boxes_in_pixels = []
    for box in boxes:
        if box.get_label() == 0:
            xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(box,
                                                                   image_w=image_w,
                                                                   image_h=image_h)
            boxes_in_pixels.append([xmin, xmax, ymin, ymax])

    tooth_tips_x = [box[0] + (box[1] - box[0]) / 2 for box in boxes_in_pixels]
    tooth_tips_x = sorted(tooth_tips_x)
    inter_tooth_dists_x = [x2 - x1 for x1, x2 in
                           zip(tooth_tips_x[:-1], tooth_tips_x[1:])]
    if inter_tooth_dists_x:
        median_intertooth_dist_x = statistics.median(inter_tooth_dists_x)
    else:  # if no inter_tooth_dists, i.e. only a single tooth
        median_intertooth_dist_x = 40.

    x_margin = int(x_factor * median_intertooth_dist_x)
    y_top_margin = int(median_intertooth_dist_x)
    y_bottom_margin = int(y_bottom_factor * median_intertooth_dist_x)

    return x_margin, y_top_margin, y_bottom_margin

def get_patch_margins_toothline(boxes, num_teeth, image_w=640, image_h=640,
                                x_factor=1.3, y_bottom_factor=3.):
    """
    Selects margins in pixels to select for detections.
    Divides toothline width by the number of teeth and uses this distance.
    :param boxes:
    :param num_teeth:
    :param image_w:
    :param image_h:
    :param x_factor:
    :param y_bottom_factor:
    :return:
    """
    is_toothline = False
    for box in boxes:
        if box.get_label() == 1:
            is_toothline = True
            xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(box,
                                                                   image_w=image_w,
                                                                   image_h=image_h)
    if is_toothline:
        mean_intertooth_dist_x = (xmax - xmin) / num_teeth
        pass
    else:
        mean_intertooth_dist_x = 40.

    x_margin = int(x_factor * mean_intertooth_dist_x)
    y_top_margin = int(mean_intertooth_dist_x)
    y_bottom_margin = int(y_bottom_factor * mean_intertooth_dist_x)

    return x_margin, y_top_margin, y_bottom_margin

def get_tooth_wm_patches(image, boxes, patch_h, patch_w, num_teeth,
                         patch_selection_method="toothline"):
    """ Get patches of objects where each patch should include a tooth and
    corresponding WM landmarks below it.
    Returns:
        patch_coords:list are coords of the patch in the full image
        coord_scales:list in order to map the keypoint coordinates back after resizing
        tooth_tips_from_det:list is center of tooth detection, used for post-processing
    """
    image_h, image_w, _ = image.shape
    patches = []
    patch_coords = []
    coord_scales = []
    tooth_tips_from_det = []
    if patch_selection_method == "intertooth":
        x_margin, y_top_margin, y_bottom_margin = get_patch_margins_intertooth(
            boxes, image_w=image_w, image_h=image_h)
    elif patch_selection_method == "toothline":
        x_margin, y_top_margin, y_bottom_margin = get_patch_margins_toothline(
            boxes, num_teeth, image_w=image_w, image_h=image_h)
    else:
        raise Exception("wrong patch selection method for keypoints")

    for box in boxes:
        if box.get_label() == 0:
            xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(box, image_w=image_w,
                                                                   image_h=image_h)
            x_center = xmin + (xmax - xmin) / 2
            y_center = ymin + (ymax - ymin) / 2
            xmin = x_center - x_margin
            ymin = y_center - y_top_margin
            xmax = x_center + x_margin
            ymax = y_center + y_bottom_margin
            xmin = max([0, xmin])
            ymin = max([0, ymin])
            patch = image[int(ymin):int(ymax), int(xmin):int(xmax)]
            x_scale = patch.shape[1] / patch_w
            y_scale = patch.shape[0] / patch_h
            patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
            patches.append(patch)
            patch_coords.append((xmin, xmax, ymin, ymax))
            coord_scales.append((x_scale, y_scale))

            tooth_tip_from_det = [(xmax - xmin) / (2 * x_scale),
                                  (ymax - ymin) / (2 * y_scale)]
            tooth_tips_from_det.append(tooth_tip_from_det)

    return patches, patch_coords, coord_scales, tooth_tips_from_det

def map_keypoints(keypoints, patch_coords, coord_scales):
    """ Map predicted keypoints from patch coords to the original image coordinates"""
    image_mapped_keypoints = []
    for patch_keypoints, patch_coord, scale in zip(keypoints, patch_coords, coord_scales):
        patch_mapped_keypoints = []
        if patch_keypoints != []:
            for i, joint_keypoint in enumerate(patch_keypoints):
                x, y = joint_keypoint
                # if x is None or y is None: x, y = 0, 0
                if x == 0 and y == 0:
                    x, y = 0, 0
                else:
                    x *= scale[0]
                    y *= scale[1]
                    x += patch_coord[0]
                    y += patch_coord[2]
                    x, y = int(x), int(y)
                patch_mapped_keypoints.append([x, y])
        image_mapped_keypoints.append(patch_mapped_keypoints)

    return image_mapped_keypoints

def draw_keypoints(image, keypoints, circle_radius=3):
    """ Draw keypoints on the full image."""
    colors = [[128, 0, 128], [0, 255, 0], [0, 0, 255], [255, 0, 0], [153, 255, 255],
              [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    #print('\ndraw_keypoints')
    #print(keypoints)
    #print('\n\n')

    for patch_keypoints in keypoints:
        if patch_keypoints == []: continue
        for i, joint_keypoint in enumerate(patch_keypoints):
            x, y = joint_keypoint
            if x == 0 or y == 0:
                continue
            cv2.circle(image, (x, y), circle_radius, colors[i], -1)

    return image

def get_pose_session(tf_model_path):
    """ Returns session (which is coupled with a Graph) and x (which is feed_dict) and y
    (which is fetches)."""
    tf_model = load_graph(tf_model_path)
    x = tf_model.get_tensor_by_name('prefix/input_image:0')
    y = tf_model.get_tensor_by_name('prefix/Add_8:0')
    gpu_options = tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.3)
    sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                 use_per_session_threads=False)
    tf_session = Session(graph=tf_model, config=sess_config)

    return tf_session, x, y

    

# keypoint detection lib core inference
def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def predict_on_image(image, model, threshold=0.3):
    """ Uses pytorch's Pose-Resnet to predict a map and then decodes it.
    Uses a `threshold` on the values of the map."""
    model.eval()
    _, h, w = image.shape
    image = image[None, ...]  # batch_size of 1
    with torch.no_grad():
        output = model(image)
        preds, maxvals = get_max_preds(output.cpu().numpy())
        preds, maxvals = preds[0], maxvals[0]  # because batch_size of 1
        # Times by 4, because output map resolution is decreased by
        # a factor of with respect to input image
        preds = preds * 4

    for i, maxval in enumerate(maxvals):
        if maxval < threshold:
            preds[i] = [0, 0]

    return preds.tolist()

def predict_on_images(images, model):
    all_keypoints = []
    for image in images:
        image = transf.ToTensor()(image).cuda()
        single_image_keypoints = predict_on_image(image, model)
        all_keypoints.append(single_image_keypoints)

    return all_keypoints

def tf_predict_on_image(image, sess, x, y, image_w, image_h, threshold=0.3):
    image = cv2.resize(image, (image_w, image_h))
    image = image / 255.
    image = image.transpose((2, 0, 1))
    image = image[None, ...]
    output = sess.run(y, {x: image})

    preds, maxvals = get_max_preds(output)
    preds, maxvals = preds[0], maxvals[0]  # because batch_size of 1
    # Times by 4, because output map resolution is decreased by
    # a factor of 4 with respect to input image
    preds = preds * 4

    for i, maxval in enumerate(maxvals):
        if maxval < threshold:
            preds[i] = [0, 0]

    return preds.tolist()

def tf_predict_on_images(images, sess, x, y, image_w, image_h):
    all_keypoints = []
    for image in images:
        keypoints = tf_predict_on_image(image, sess, x, y, image_w, image_h)
        all_keypoints.append(keypoints)

    return all_keypoints




class WmNetRunner(object):
    def __init__(self, keypointsWeights, yoloWeights=None, ssdPredsDir=None):
        self.obj_thresh = 0.2
        self.class_obj_threshold = [0.5, 0.5, 0.4, 0.4, 0.4]
        self.nms_threshold=0.01
        self.is_filter_bboxes = False
        self.shovel_type = "Cable"
        self.num_teeth = 6
        self.input_size = 640
        self.labels = ["Tooth", "Toothline", "BucketBB", "MatInside", "WearArea"]
        self.max_box_per_image = 20
        self.anchors = [1.27,2.0,  0.83,1.24,  0.48,0.54,  0.60,1.3, 7.30,1.60,  7.23,3.13,  8.96,4.44,   13.47,9.82,  11.32,7.42]

        self.patch_w, self.patch_h = 96, 192


        if yoloWeights:
            ###############################
            #   Load the yolo model 
            self.yolo = YOLO(
                input_size         = self.input_size, 
                labels             = self.labels, 
                max_box_per_image  = self.max_box_per_image,
                anchors            = self.anchors)

            self.yolo.load_weights(yoloWeights)
            ###############################


        if ssdPredsDir:
            self.ssdPredsDir = ssdPredsDir


        ###############################
        #   Load the keypoint detection model         
        self.tf_session, self.x, self.y = get_pose_session(keypointsWeights)
        ###############################

    def yoloPredict(self, inImage):
        boxes, filtered_boxes = self.yolo.predict(
            inImage,
            obj_threshold=self.obj_thresh,
            nms_threshold=self.nms_threshold,
            is_filter_bboxes=self.is_filter_bboxes,
            shovel_type=self.shovel_type,
            class_obj_threshold=self.class_obj_threshold
        )

        boxes += filtered_boxes


        #visualize the predictions
        image = draw_boxes(inImage, boxes, labels=self.labels, score_threshold=self.obj_thresh) 

        return boxes, image

    def keypointsPredict(self, inImage, boxes):
        patches, patch_coords, coord_scales, tooth_tips_from_det = get_tooth_wm_patches(
            inImage,
            boxes,
            self.patch_h,
            self.patch_w,
            self.num_teeth)

        keypoints = tf_predict_on_images(patches, self.tf_session, self.x, self.y, self.patch_w, self.patch_h)

        mapped_keypoints = map_keypoints(keypoints, patch_coords, coord_scales)

        outImageKeypoints = draw_keypoints(inImage, mapped_keypoints, circle_radius=3)

        return mapped_keypoints, outImageKeypoints

    def getBoxesFromSsdPreds(self, jsonName, inImage):
        boxes = []

        with open(self.ssdPredsDir + jsonName, 'r') as fjson:
            data = tuple(json.load(fjson))
            
            for el in data:

                if el[4] == 'teethLine':
                    aBox = BoundBox(el[0], el[2], el[1], el[3], c=1, classes=[0, 1.0, 0, 0, 0], label=1, score= el[3])

                if el[4] == 'tooth':
                    aBox = BoundBox(el[0], el[2], el[1], el[3], c=0, classes=[1.0, 0, 0, 0, 0], label=0, score= el[3])
                    
                boxes.append(aBox)


        image = draw_boxes(inImage, boxes, labels=self.labels, score_threshold=self.obj_thresh) 

        return boxes, image