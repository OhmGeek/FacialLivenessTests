from abc import ABC, abstractmethod

class AbstractMetricVectorCreator(ABC):
    def __init__(self, metrics):
        self.metrics = metrics

    @abstractmethod
    def create_vector(self, input, blurred_image):
        pass


class DefaultMetricVectorCreator(AbstractMetricVectorCreator):
    def create_vector(self, input, blurred_image):
        metric_vector = []
        for m in self.metrics:
            value = m.calculate(input, blurred_image)
            metric_vector.append(value)

        return metric_vector