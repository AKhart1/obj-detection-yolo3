import cv2
import numpy as np
from core.entities.detected_object import DetectedObject
from core.interfaces.detector import IDetector
from utils.config_loader import load_yolo_config

class OpenCVDetector(IDetector):
    def __init__(self, config_path, weights_path, classes_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = self.load_classes(classes_path)
        self.COLORS = np.array([
            [0, 0, 255],  # Red (BGR format)
            [0, 255, 0],  # Green
            [255, 0, 0],  # Blue
            [0, 255, 255],  # Yellow
            [255, 0, 255],  # Magenta
            [255, 255, 0],  # Cyan
        ])

        if len(self.classes) > len(self.COLORS):
            extra_colors = np.random.uniform(0, 255, size=(len(self.classes) - len(self.COLORS), 3))
            self.COLORS = np.vstack([self.COLORS, extra_colors])

        self.conf_threshold = 0.5
        self.nms_threshold = 0.4

    def load_classes(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        try:
            return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            return [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, image):
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        detected_objects = []

        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]
            x = round(box[0])
            y = round(box[1])
            w = round(box[2])
            h = round(box[3])
            detected_objects.append(DetectedObject(class_ids[i], confidences[i], x, y, x + w, y + h))

        return detected_objects

    def draw_predictions(self, image, detected_objects):
        for obj in detected_objects:
            label = str(self.classes[obj.class_id])
            color = self.COLORS[obj.class_id].tolist()  # Convert to list for cv2

            cv2.rectangle(image, (obj.x, obj.y), (obj.x_plus_w, obj.y_plus_h), color, 2)

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            label_x = obj.x
            label_y = obj.y - label_size[1] - 5
            cv2.rectangle(image, (label_x, label_y), (label_x + label_size[0], label_y + label_size[1] + 5), color, cv2.FILLED)
            
            cv2.putText(image, label, (label_x, label_y + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)            
