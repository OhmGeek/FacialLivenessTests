""" JPEG Quality Index """

from liveness.quality.metrics.generic import AbstractNoReferenceQualityMetric
import numpy as np

class JPEGQualityIndexMetric(AbstractNoReferenceQualityMetric):
    def __init__(self, logger, alpha=-245.8909, beta=261.9373, gamma1=-239.8886, gamma2=160.1664, gamma3=64.2859):
        self._alpha = alpha
        self._beta = beta
        self._gamma1 = gamma1
        self._gamma2 = gamma2
        self._gamma3 = gamma3
        super().__init__(logger)

    def calculate(self, image):
        M, N = image.shape # get shape.

        # Horizontal Features
        d = image[:, 2:N] - image[:, 1:(N - 1)]

        B_h = np.mean(np.abs(d[:, 8:8:8*floor(N/8) - 1]))

        A_h = (8 * np.mean(np.abs(d)) - B) / 7
        
        sig = np.sign(d)

        left_sig = sig[:, 1:(N-2)]
        right_sig = sig[:, 2:(N-1)]

        Z_h = np.mean(np.multiply(left_sig, right_sig) < 0)

        # Vertical Features
        d = image[:, 2:N] - image[:, 1:(N - 1)]

        B_v = np.mean(np.abs(d[:, 8:8:8*floor(N/8) - 1]))

        A_v = (8 * np.mean(np.abs(d)) - B) / 7
        
        sig = np.sign(d)

        left_sig = sig[:, 1:(N-2)]
        right_sig = sig[:, 2:(N-1)]

        Z_v = np.mean(np.multiply(left_sig, right_sig) < 0)

        # Combined features
        B = (B_h + B_v) / 2
        A = (A_h + A_v) / 2
        Z = (Z_h + Z_v) / 2

        # Score calculation
        score = self._alpha + self._beta* (np.power(B, self._gamma1 / 10000)) * (np.power(A, self._gamma2 / 10000)) * (np.power(Z, self._gamma3 / 10000))
        
        return score

