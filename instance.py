import pixellib
from pixellib.instance import instance_segmentation
import time

segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")
start = time.perf_counter()
segment_image.segmentImage("images/UWTEST.jpg", output_image_name = "maskRCNNUWTEST.jpg")
time_took = time.perf_counter() - start
print(f"Time took: {time_took:.2f}s")