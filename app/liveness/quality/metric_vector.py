from abc import ABC, abstractmethod
from liveness.quality.metrics.generic import AbstractQualityMetric, AbstractNoReferenceQualityMetric
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
            # TODO: refactor this, as this is dumb.
            value = None
            print("Looking at metric: ", m)
            if(isinstance(m, AbstractNoReferenceQualityMetric)):
                # if no reference, just pass input.

                value = m.calculate(input)
                print(value)
            elif(isinstance(m, AbstractQualityMetric)):
                # if abstract quality metric, pass both images
                
                value = m.calculate(input, blurred_image)

            if value is not None:
                metric_vector.append(value)

        return metric_vector