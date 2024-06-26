import cv2
import numpy
import serial
import time
import RPi.GPIO as GPIO
from time import sleep

# for GPIO numbering, choose BCM  
# GPIO.setmode(GPIO.BCM)  


# Pins for Motor Driver Inputs
Motor1A = 24
Motor1B = 23
Motor1E = 25
LED = 14

def setup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)              # GPIO Numbering
    GPIO.setup(Motor1A,GPIO.OUT)  # All pins as Outputs
    GPIO.setup(Motor1B,GPIO.OUT)
    GPIO.setup(Motor1E,GPIO.OUT)
    GPIO.setup(LED,GPIO.OUT,initial=GPIO.LOW)

def forwards():
    # Going forwards
    GPIO.output(Motor1A,GPIO.HIGH)
    GPIO.output(Motor1B,GPIO.LOW)
    GPIO.output(Motor1E,GPIO.HIGH)
    print("Train Running")
 
def fstop():
    # fstop
    GPIO.output(Motor1E,GPIO.LOW)
    GPIO.output(Motor1A,GPIO.LOW)
    print("Stopping the train")
   
   
classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                GPIO.output(LED,GPIO.HIGH)
                fstop()
                                                           
                if (draw):
                   
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                   
            else:
                forwards()  
                GPIO.output(LED,GPIO.LOW)
    return img,objectInfo


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    setup()
    forwards()
    cap.set(3,640)
    cap.set(4,480)


    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.4,0.2,objects=['elephant'])
        cv2.imshow("Output",img)
        cv2.waitKey(1)
