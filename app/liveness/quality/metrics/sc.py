"""
Structural Content
"""
import numpy as np

from liveness.quality.metrics.generic import AbstractQualityMetric


class StructuralContentMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        image_squared_sum = np.square(image).sum()
        blurred_image_squared_sum = np.square(blurred_image).sum()

        return image_squared_sum / blurred_image_squared_sum
