"""
Normalised Absolute Error
"""
import numpy as np

from metrics.generic import AbstractQualityMetric


class NormalisedAbsoluteErrorMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        # First, calculate I - G
        numerator = np.subtract(image, blurred_image)
        numerator = np.absolute(numerator)
        # Sum all the values together
        numerator = numerator.sum()

        # Sum the real image absolute values.
        denominator = np.absolute(image).sum()

        output = numerator / denominator

        return output
