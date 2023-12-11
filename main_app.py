import streamlit as st
import cv2
import numpy as np

st.write("# Object Detection using YOLOV3")


st.markdown("Final Exam: Model Deployment in the Cloud") 
st.text("Emmanuel Villanueva")
st.text("Michael Vincent R. Alcoseba")
st.text("John Terah Saquitan")

st.text("CPE 019-CPE32S2 - Emerging Technologies 2 in CpE")

classes_file = 'coco.names'

# load class names
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

file = st.file_uploader("Choose photo from computer", type=["jpg", "png", "jpeg"])

classes_file = 'coco.names'
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

if file is None:
    st.text("Please upload an image file")
else:
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Preprocess the image by resizing and normalizing
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Extract the bounding box coordinates
                box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                # Draw the bounding box and label on the image
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                label = classes[class_id]
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(image, use_column_width=True, channels="BGR")
