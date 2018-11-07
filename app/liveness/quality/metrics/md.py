""" Maximum Difference """
import numpy as np

from metrics.generic import AbstractQualityMetric


class MaximumDifferenceMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate maximum difference in the image
        :param image: full quality image
        :param blurred_image: gaussian blurred quality image
        :return: the maximum difference
        """
        difference = np.subtract(image, blurred_image)
        difference = np.absolute(difference)

        return difference.max()
