"""
Gradient Magnitude Error Metric
"""
from liveness.quality.metrics.generic import AbstractQualityMetric
import numpy as np
import cv2

class GradientMagnitudeErrorMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate the gradient magnitude metric on an image and it's corresponding
        gaussian blurred image
        :param image: the full quality image
        :param blurred_image: the gaussian blurred image
        :return: the average difference value
        """

        image_grad = cv2.Sobel(image,cv2.CV_64F,1,1,ksize=5)

        blurred_image_grad = cv2.Sobel(blurred_image,cv2.CV_64F,1,1,ksize=5)

        return np.mean(np.square(np.subtract(image_grad, blurred_image_grad)))
