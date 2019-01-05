"""
Spectral Phase Error
"""
from liveness.quality.metrics.generic import AbstractQualityMetric
import numpy as np
import cv2

class SpectralPhaseErrorMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate the spectral phase metric on an image and it's corresponding
        gaussian blurred image
        :param image: the full quality image
        :param blurred_image: the gaussian blurred image
        :return: the average difference value
        """

        # Run DFT. Two channels output: one being real, one being complex plane.
        fourier_image = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
        fourier_blurred_image = cv2.dft(np.float32(blurred_image), flags = cv2.DFT_COMPLEX_OUTPUT)
        
        image_dft_shift = np.fft.fftshift(fourier_image)
        image_dft_shift_x = image_dft_shift[:,:, 0]
        image_dft_shift_y = image_dft_shift[:, :, 1]

        blurred_image_dft_shift = np.fft.fftshift(fourier_blurred_image)
        blurred_image_dft_shift_x = blurred_image_dft_shift[:,:,0]
        blurred_image_dft_shift_y = blurred_image_dft_shift[:,:,1]

        image_mag = cv2.magnitude(image_dft_shift_x, image_dft_shift_y)
        blurred_image_mag = cv2.magnitude(blurred_image_dft_shift_x, blurred_image_dft_shift_y)


        return np.mean(np.square(np.subtract(image_mag, blurred_image_mag)))
        

