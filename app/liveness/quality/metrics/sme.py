"""
Spectral Magnitude Error
"""
from liveness.quality.metrics.generic import AbstractQualityMetric
import numpy as np
import cv2

class SpectralMagnitudeErrorMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate the spectral magnitude metric on an image and it's corresponding
        gaussian blurred image
        :param image: the full quality image
        :param blurred_image: the gaussian blurred image
        :return: the average difference value
        """

        # Run DFT. Two channels output: one being real, one being complex plane.
        fourier_image = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
        fourier_blurred_image = cv2.dft(np.float32(blurred_image), flags = cv2.DFT_COMPLEX_OUTPUT)
        
        image_dft_shift = np.fft.fftshift(fourier_image)
        blurred_image_dft_shift = np.fft.fftshift(fourier_blurred_image)

        fourier_mag_image = 20*np.log(cv2.magnitude(image_dft_shift[:,:,0],image_dft_shift[:,:,1]))
        fourier_mag_blurred_image = 20*np.log(cv2.magnitude(blurred_image_dft_shift[:,:,0],blurred_image_dft_shift[:,:,1]))
        
        return np.mean(np.square(np.subtract(fourier_mag_image, fourier_mag_blurred_image)))
        

