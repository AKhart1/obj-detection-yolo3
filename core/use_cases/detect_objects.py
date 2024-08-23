class DetectObjectsUseCase:
    def __init__(self, detector):
        self.detector = detector

    def execute(self, image):
        return self.detector.detect(image)
