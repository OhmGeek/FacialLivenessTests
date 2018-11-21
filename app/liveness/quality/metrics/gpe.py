"""
Gradient Phase Error Metric
"""
from liveness.quality.metrics.generic import AbstractQualityMetric
import numpy as np
import cv2

class GradientPhaseErrorMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate the gradient phase metric on an image and it's corresponding
        gaussian blurred image
        :param image: the full quality image
        :param blurred_image: the gaussian blurred image
        :return: the average difference value
        """
        image_grad_x = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
        image_grad_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

        blurred_image_grad_x = cv2.Sobel(blurred_image,cv2.CV_64F,1,0,ksize=5)
        blurred_image_grad_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

        image_mag = cv2.magnitude(image_grad_x, image_grad_y)
        blurred_image_mag = cv2.magnitude(blurred_image_grad_x, blurred_image_grad_y)

        return np.mean(np.square(np.subtract(image_mag, blurred_image_mag)))
        

