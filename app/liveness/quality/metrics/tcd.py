"""
Total Corner Difference
"""
from liveness.quality.metrics.generic import AbstractQualityMetric
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

def get_number_of_corners(image, thresh=255):
   # Detector parameters
    counter = 0
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    dst = cv2.cornerHarris(image, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                counter += 1

    return counter

class TotalCornerDifferenceMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate the total corner difference metric on an image and it's corresponding
        gaussian blurred image
        :param image: the full quality image
        :param blurred_image: the gaussian blurred image
        :return: the average difference value
        """
        threshold = 160
        num_corners_image = get_number_of_corners(image, thresh=threshold)
        num_corners_blurred_image = get_number_of_corners(blurred_image, thresh=threshold)

        return np.abs(num_corners_image - num_corners_blurred_image) / max([num_corners_image, num_corners_blurred_image]) 


        
