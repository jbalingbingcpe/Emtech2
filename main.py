import os.path
import cv2
import numpy as np
import requests

yolo_config = 'yolov3-tiny.cfg'
yolo_weights = 'yolov3-tiny.weights'
classes_file = 'coco.names'

with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

image = cv2.imread('test.jpg')
cv2.imshow('Image', image)
cv2.waitKey(0)

net = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
print(output_layers)
outputs = net.forward(output_layers)

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (center_x, center_y, width, height) = box.astype("int")
            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))

            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            label = classes[class_id]
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)