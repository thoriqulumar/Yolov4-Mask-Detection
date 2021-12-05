import cv2 
import os
import numpy as np

CONFIDENCE_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

CONFIG_PATH = "D:/Programming/Python/realtime detection flask/web/yolov4-tiny-custom.cfg"
WEIGHTS_PATH = "D:/Programming/Python/realtime detection flask/web/yolov4-tiny-custom_last.weights"
LABELS = open('web\model\obj.names').read().strip().split('\n')
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

img_path = 'D:/Programming/Python/realtime detection flask/web/test/mask_people.jpg'
image = cv2.imread(img_path)
file_name = os.path.basename(img_path)
filename, ext = file_name.split(".")

img_blob = cv2.dnn.blobFromImage(image, 1/255, (416,416), swapRB=True, crop=False)

net.setInput(img_blob)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

layer_outputs = net.forward(ln)

font_scale = 1
thickness = 1
boxes, confidences, class_ids = [], [], []

for output in layer_outputs:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence_score = score[class_id]
        if confidence_score>CONFIDENCE_THRESHOLD:
            bbox = detection[:4]
            x_center, y_center, width, height = bbox
            x = int(x_center - (width/2))
            y = int(y_center - (height/2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence_score))
            class_ids.append(class_ids)

for i in range(len(boxes)):
    # extract the bounding box coordinates
    x, y = boxes[i][0], boxes[i][1]
    w, h = boxes[i][2], boxes[i][3]
    # draw a bounding box rectangle and label on the image
    color = [int(c) for c in COLORS[class_ids[i]]]
    cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
    text = f"{LABELS[class_ids[i]]}: {confidences[i]:.2f}"
    # calculate text width & height to draw the transparent boxes as background of the text
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
    text_offset_x = x
    text_offset_y = y - 5
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
    overlay = image.copy()
    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
    # add opacity (transparency to the box)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    # now put the text (label: confidence %)
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
            
cv2.imwrite(filename + "_predicted." + ext, image)