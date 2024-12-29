from LLBMA.brain.BMAYOLOManager import YOLO_detect
from LLBMA.resources.BMAassumptions import YOLO_ckpt_path, YOLO_conf_thres
from ultralytics import YOLO
from PIL import Image

image_path = "3316.jpg"
image = Image.open(image_path)

model = YOLO(YOLO_ckpt_path)

results = YOLO_detect(model, image, conf_thres=YOLO_conf_thres)

print(results)
