"""
Laplacian Mean Squared Error metric.

"""
from app.liveness.quality.metrics.generic import AbstractQualityMetric


class LaplacianMeanSquaredMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        raise NotImplementedError()