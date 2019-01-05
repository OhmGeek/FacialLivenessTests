"""
R-Average MD
"""
import numpy as np

from liveness.quality.metrics.generic import AbstractQualityMetric


class RAveragedMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        R = 10 # The default R value (todo refactor into constructor)

        abs_difference = np.absolute(np.subtract(image, blurred_image))

        # Sum all values smaller than or equal to R
        values_to_sum = abs_difference[np.where(abs_difference <= R)]
        
        return sum(values_to_sum) / R

