from abc import abstractmethod, ABC


class AbstractQualityMetric(ABC):
    def __init__(self, logger):
        self._logger = logger

    @abstractmethod
    def calculate(self, image, blurred_image):
        pass