""" High Low Frequency index Index """

from liveness.quality.metrics.generic import AbstractNoReferenceQualityMetric
import numpy as np
import math
import cv2
class HighLowFrequencyIndexMetric(AbstractNoReferenceQualityMetric):
    def calculate(self, image):
        i_l = 0.15 * image.shape[0] 
        i_h = 0.15 * image.shape[0]
        j_l = 0.15 * image.shape[1]
        j_h = 0.15 * image.shape[1]

        dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

        denominator = np.abs(magnitude_spectrum).sum()
        print(denominator)
        # First part of numerator: 1 to il, and jl.
        low_freq_sum = 0
        for i in range(0,math.ceil(i_l)):
            for j in range(0, math.ceil(j_l)):
                low_freq_sum += magnitude_spectrum[i, j]
        
        # Now second part
        high_freq_sum = 0
        for i in range(math.ceil(i_h + 1), image.shape[0]):
            for j in range(math.ceil(j_h + 1), image.shape[1]):
                high_freq_sum += magnitude_spectrum[i, j]

        numerator = low_freq_sum - high_freq_sum

        return numerator / denominator



