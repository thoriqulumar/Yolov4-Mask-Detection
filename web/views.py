from flask import Flask, render_template, Blueprint, Response
import cv2
import numpy as np
from . import utils

views = Blueprint('views', __name__)
camera = cv2.VideoCapture(0)

def load_model():
    net = cv2.dnn.readNetFromDarknet(utils.CONFIG_PATH,  utils.WEIGHTS_PATH)
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, layer_names

def convert_bbox(bbox):
    center_x, center_y, width, height = bbox
    x = int(center_x - (width/2))
    y = int(center_y - (height/2))
    return x, y, int(width), int(height)

def start_video():
    net, layers_name = load_model()
    while True:
        _, image = camera.read()
        flip_image = cv2.flip(image, 1)
        h, w = flip_image.shape[:2]
        blob = cv2.dnn.blobFromImage(flip_image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(layers_name)
        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
            
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > utils.CONFIDENCE_THRESHOLD:
                
                    box = detection[:4] * np.array([w, h, w, h])
                    x, y, width, height = convert_bbox(box)

                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        nms_bounding_box = cv2.dnn.NMSBoxes(boxes, confidences, utils.SCORE_THRESHOLD, utils.IOU_THRESHOLD)

        if len(nms_bounding_box) > 0:
            for i in nms_bounding_box.flatten():
                x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
               
                color = [int(c) for c in utils.COLORS[class_ids[i]]]
                cv2.rectangle(flip_image, (x, y), (x + w, y + h), color=color, thickness=utils.THICKNESS)
                text = f"{utils.LABELS[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.putText(flip_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=utils.FONT_SCALE, color=(255, 255, 255), thickness=utils.THICKNESS)



        ret, buffer = cv2.imencode('.jpg', flip_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@views.route('/stream')
def stream():
    return Response(start_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@views.route('/')
def main():
    return render_template('index.html')