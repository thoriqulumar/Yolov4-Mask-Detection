import numpy as np


CONFIDENCE_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


CONFIG_PATH = "D:/Programming/Python/realtime detection flask/web/model/yolov4-tiny-custom.cfg"
WEIGHTS_PATH = "D:/Programming/Python/realtime detection flask/web/model/yolov4-tiny-custom_last.weights"
LABELS = open('web\model\obj.names').read().strip().split('\n')
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

THICKNESS = 1
FONT_SCALE =1