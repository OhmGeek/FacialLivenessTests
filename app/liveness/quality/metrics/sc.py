"""
Structural Content
"""
import numpy as np

from app.liveness.quality.metrics.generic import AbstractQualityMetric


class StructuralContentMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        image_squared_sum = np.squared(image).sum()
        blurred_image_squared_sum = np.squared(blurred_image).sum()

        return image_squared_sum / blurred_image_squared_sum
