"""
Mean Squared Error
"""
from app.liveness.quality.metrics.generic import AbstractQualityMetric
import numpy as np


class MeanSquaredErrorMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate the mean squared error metric on an image and it's corresponding
        gaussian blurred image
        :param image: the full quality image
        :param blurred_image: the gaussian blurred image
        :return: the mse value
        """

        mse = np.square(np.subtract(image, blurred_image)).mean()

        return mse
