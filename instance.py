import pixellib
from pixellib.instance import instance_segmentation
import cv2

instance_seg = instance_segmentation()
instance_seg.load_model("mask_rcnn_coco.h5")
segmask, output = instance_seg.segmentImage("UWTEST.jpg", show_bboxes= True)
cv2.imwrite("img.jpg", output)
print(output.shape)