import subprocess
import cv2
import time
import numpy as np
import pyttsx3

engine = pyttsx3.init()
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)                        #printing current voice rate
engine.setProperty('rate', 100)
voices = engine.getProperty('voices') 
engine.setProperty('voice', voices[0].id)   #changing index, changes voices. 1 for female
engine.say("VISION TURNING ON")
engine.runAndWait()

def textToSpeech(txt):
    engine = pyttsx3.init()
    engine.getProperty('rate')
    engine.setProperty('rate', 100)   
    engine.say(txt)
    engine.runAndWait()


thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)

#cap.set(3,1280)
#cap.set(4,720)
cap.set(10,150)
classNames= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
while True:
    success,img = cap.read()
    img=cv2.flip(img,1)
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    texts=[]
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[1]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        print(classNames[classIds[i][0]-1].upper())
    cv2.imshow("Output",img)
    cv2.waitKey(1)
    #textToSpeech(classNames[classIds[i][0]-1])
    more_lines = ["",(classNames[classIds[i][0]-1]+' detected')]
    with open('readme.txt', 'a') as f:
        f.writelines('\n'.join(more_lines))
        #time.sleep(2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
