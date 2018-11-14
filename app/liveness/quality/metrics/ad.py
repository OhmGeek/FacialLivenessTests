"""
Average Difference
"""
from metrics.generic import AbstractQualityMetric
import numpy as np


class AverageDifferenceMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate the average difference metric on an image and it's corresponding
        gaussian blurred image
        :param image: the full quality image
        :param blurred_image: the gaussian blurred image
        :return: the average difference value
        """

        difference = image - blurred_image
        average_difference = np.mean(difference)

        return average_difference