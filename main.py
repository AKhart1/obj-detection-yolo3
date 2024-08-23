import cv2
from utils.config_loader import load_yolo_config
from utils.image_processor import load_image
from data.opencv_detector import OpenCVDetector
from core.use_cases.detect_objects import DetectObjectsUseCase

def main():
    image_path, config_path, weights_path, classes_path = load_yolo_config()
    
    detector = OpenCVDetector(config_path, weights_path, classes_path)
    detect_objects_use_case = DetectObjectsUseCase(detector)
    
    image = load_image(image_path)
    detected_objects = detect_objects_use_case.execute(image)
    
    detector.draw_predictions(image, detected_objects)
    
    cv2.imshow("object detection", image)
    cv2.waitKey()
    cv2.imwrite("object_detected.jpg", image)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
