"""
Total Edge Difference
"""
from metrics.generic import AbstractQualityMetric
import numpy as np
import cv2

# In order to implement both features, which are computed
# according to the corresponding expressions given in
# Table I, we use: (i) the Sobel operator to build the binary
# edge maps IE and ˆ
# IE ; (ii) the Harris corner detector [48]
# to compute the number of corners Ncr and Nˆcr found in
# I and ˆ
# I.
class TotalEdgeDifferenceMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate the total edge difference metric on an image and it's corresponding
        gaussian blurred image
        :param image: the full quality image
        :param blurred_image: the gaussian blurred image
        :return: the average difference value
        """
        # First, use sobel.
        image_grad = cv2.Sobel(image,cv2.CV_64F,1,1,ksize=5)
        blurred_image = cv2.Sobel(blurred_image,cv2.CV_64F,1,1,ksize=5)

        np.subtract(image_grad, blurred_image)

        return np.mean(np.abs(np.subtract(image_grad, blurred_image_grad)))


        
