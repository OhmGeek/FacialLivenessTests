""" Signal to noise ratio"""
import numpy as np

from liveness.quality.metrics.generic import AbstractQualityMetric
from liveness.quality.metrics.mse import MeanSquaredErrorMetric


class SignalToNoiseRatioMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        mse_helper = MeanSquaredErrorMetric(self._logger)
        numerator = np.square(image).sum()

        # N * M * MSE(I, G)
        denominator = image.shape[0] * image.shape[1]
        denominator *= mse_helper.calculate(image, blurred_image)

        output = 10 * np.log(numerator / denominator)

        return output