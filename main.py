import cv2
import numpy as np

import time
import sys
import os

# This application of YOLO is mainly inspired by https://www.thepythoncode.com/article/yolo-object-detection-with-opencv-and-pytorch-in-python
CONFIDENCE = 0.5
SCORE_THRESH = 0.5
IOU_THRESH = 0.5

# the neural network configuration
config_path = "cfg/yolov3.cfg"
# the YOLO net weights file
weights_path = "weights/yolov3.weights"

# loading all the class labels (objects)
labels = open("data/coco.names").read().strip().split("\n")
# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# image prep
# path_name = "images/street.jpg"
path_name = input("Enter picture path name:")
# user's desired object of choice
desired = input("Enter desired object:")
# user's desired way of processing
process = input("Enter desired processing way(m/b/s):")
# if it's putting a sticker, enter sticker name
if process == 's':
    sec_path_name = input("Enter pathname for sticker(needs to be png file):")

image = cv2.imread(path_name)
file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")

h, w = image.shape[:2]
# create 4D blob
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

print("image.shape:", image.shape)
print("blob.shape:", blob.shape)

# sets the blob as the input of the network
net.setInput(blob)
# get all the layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# feed forward (inference) and get the network output
# measure how much it took in seconds
start = time.perf_counter()
layer_outputs = net.forward(ln)
time_took = time.perf_counter() - start
print(f"Time took: {time_took:.2f}s")

font_scale = 1
thickness = 1
boxes, confidences, class_ids = [], [], []
# loop over each of the layer outputs
for output in layer_outputs:
    # loop over each of the object detections
    for detection in output:
        # extract the class id (label) and confidence (as a probability) of
        # the current object detection
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # discard out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > CONFIDENCE:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print(detection.shape)

# now we draw the object out

cv2.imwrite(filename + "_yolo3." + ext, image)
# perform the non maximum suppression given the scores defined before
idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESH, IOU_THRESH)

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]

        text = f"{labels[class_ids[i]]}"
        # calculate text width & height to draw the transparent boxes as background of the text
        # if the object desired by the user is found on the image
        if (text == desired):
            # if the user wishes to mark out the object
            # the program will draw a bounding box rectangle and label on the image
            if (process == "m"):
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                (text_width, text_height) = \
                    cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = (
                (text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                # add opacity (transparency to the box)
                image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                # now put the text (label: confidence %)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

            # if the user wishes to blur out the object
            # the program will blur it with gaussian filter according to the bounding box
            if (process == "b"):
                mask = np.zeros(image.shape, dtype=np.uint8)
                channel_count = image.shape[2]
                ignore_mask_color = (255,) * channel_count
                roi_corners = np.array([[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]], dtype=np.int32)
                cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask
                blurred = cv2.GaussianBlur(image, (43, 43), 30)
                image = cv2.bitwise_and(blurred, mask) + cv2.bitwise_and(image, mask_inverse)

            # if the user wishes to add a sticker on the object
            # the sticker is
            if process == "s":

                sticker = cv2.imread(sec_path_name,-1)

                # need to resize sticker according to bounding box
                sticker = cv2.resize(sticker, (w,h), interpolation = cv2.INTER_AREA)
                alpha_sticker = sticker[:, :, 3] / 255.0
                alpha_image = 1.0 - alpha_sticker
                y1 = y
                y2 = y1 + sticker.shape[0]
                x1 = x
                x2 = x1 + sticker.shape[1]
                for c in range(0,2):
                    region = image[y1:y2, x1:x2, c]
                    alpha_region = alpha_image[:region.shape[0], :region.shape[1]]
                    alpha_sticker = alpha_sticker[:region.shape[0], :region.shape[1]]
                    sticker = sticker[:region.shape[0], :region.shape[1]]
                    image[y1:y2,x1:x2,c] = (alpha_sticker*sticker[:,:,c] + alpha_region*region)

cv2.imwrite(filename + desired + process + "_yolo3." + ext, image)
cv2.imshow("image", image)
