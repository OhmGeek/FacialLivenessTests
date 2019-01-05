""" JPEG Quality Index """

from liveness.quality.metrics.generic import AbstractNoReferenceQualityMetric
import numpy as np
import math
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
        print("(M, N) = (%d, %d)" % (M, N))
        # Horizontal Features
        d = image[:, 1:(N-1)] - image[:, 0:(N - 2)]
    
        B_h = np.mean(np.abs(d[:, 8*math.floor(N/8) - 3]))

        A_h = (8 * np.mean(np.abs(d)) - B_h) / 7
        
        sig = np.sign(d)
        print(N)
        
        left_sig = sig[:, :]
        right_sig = sig[:, :]

        print(left_sig.shape, right_sig.shape)
        Z_h = np.mean(np.multiply(left_sig, right_sig))

        # Dereference for now.
        sig = None
        left_sig = None
        right_sig = None 

        # Vertical Features
        d = image[2:M, :] - image[0: (M-2), :]

        B_v = np.mean(np.abs(d[8*math.floor(M/8) - 3, :]))

        A_v = (8 * np.mean(np.abs(d)) - B_v) / 7
        

        sig = np.sign(d)
        left_sig = sig[:, :]
        right_sig = sig[:, :]

        Z_v = np.mean(np.multiply(left_sig, right_sig))

        # Combined features
        B = (B_h + B_v) / 2
        A = (A_h + A_v) / 2
        Z = (Z_h + Z_v) / 2
        print(B, A, Z)
        # Score calculation
        score = self._alpha + self._beta* (np.power(B, self._gamma1 / 10000)) * (np.power(A, self._gamma2 / 10000)) * (np.power(Z, self._gamma3 / 10000))
        return score

