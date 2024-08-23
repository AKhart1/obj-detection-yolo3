class DetectedObject:
    def __init__(self, class_id, confidence, x, y, x_plus_w, y_plus_h):
        self.class_id = class_id
        self.confidence = confidence
        self.x = x
        self.y = y
        self.x_plus_w = x_plus_w
        self.y_plus_h = y_plus_h
