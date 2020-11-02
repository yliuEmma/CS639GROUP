import cv2
import numpy as np

import time
import sys
import os

#This application of YOLO is mainly inspired by https://www.thepythoncode.com/article/yolo-object-detection-with-opencv-and-pytorch-in-python
CONFIDENCE = 0.5
SCORE_THRESH = 0.5
IOU_THRESH = 0.5

# the neural network configuration
config_path = "cfg/yolov3.cfg"
# the YOLO net weights file
weights_path = "weights/yolov3.weights"
# weights_path = "weights/yolov3-tiny.weights"

# loading all the class labels (objects)
labels = open("data/coco.names").read().strip().split("\n")
# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)