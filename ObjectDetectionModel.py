from ultralytics import YOLO
import cv2


#YOLO pretrained model with COCO dataset
model = YOLO('yolov8n.pt') #download yolo pretrained model you can change the last part to l or something else

results = model("./sampleImage/sample1.jpg",show = True)#locate the image and show it
cv2.waitKey(0)#exits when the user inputs(click the x button etc.) This is needed as the image will automaticaly close if this is not applied

