from abc import ABC, abstractmethod

class IDetector(ABC):
    @abstractmethod
    def detect(self, image):
        pass