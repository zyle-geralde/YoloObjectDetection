from ultralytics import YOLO
import cv2
import cvzone
import math

#cam = cv2.VideoCapture(0)#use to capture video
cam = cv2.VideoCapture("./sampleImage/1643-148614430_small.mp4")#for existing video

cam.set(3,1280)#set width
cam.set(4,720) #set height

model = YOLO("./YoloWeights/yolov8n.pt")

predictionNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
"diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair drier", "toothbrush"]




while True:
    success,image = cam.read()#read the image
    result = model(image,stream = True)

    for r in result:
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1),int(x2), int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),2)#create reactangle

            confidence = math.ceil((box.conf[0]*100))/100# messures how sure the model is about the detection(higher value means more certain)

            class_name = predictionNames[int(box.cls[0])]#returns the index of the label

            cvzone.putTextRect(image,f"{class_name} ",(x1,max(33,y1)),scale = 0.8,thickness=1)
    cv2.imshow("Image",image)#display image
    cv2.waitKey(1)#wait for 1 miliseconds before displaying the next image

    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:  # Window closed
        break
