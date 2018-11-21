from liveness.quality.metrics.generic import AbstractQualityMetric
import cv2
import numpy as np

class VisualInformationFidelityMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        # Declare constants
        sigma_nsq=2
        eps = 1e-10

        # Accumulators
        num = 0.0
        den = 0.0
        # Declare images
        ref = image
        dist = blurred_image
        # for 5 different scales, calculate and sum.
        for scale in range(1, 5):
            N = 2 ** (4-scale+1) + 1
            sd = N/5.0
            if(scale > 1):
                ref = cv2.GaussianBlur(image, (5,5), sd)
                dist = cv2.GaussianBlur(blurred_image, (5,5), sd)

            mu1 = cv2.GaussianBlur(ref, (5,5), sd)
            mu2 = cv2.GaussianBlur(dist, (5,5), sd)

            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = cv2.GaussianBlur(ref * ref, (5,5), sd) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(dist * dist, (5,5), sd) - mu2_sq
            sigma12 = cv2.GaussianBlur(ref * dist, (5,5), sd) - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g*sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
            sigma1_sq[sigma1_sq<eps] = 0
            
            g[sigma2_sq<eps] = 0
            sv_sq[sigma2_sq<eps] = 0
            
            sv_sq[g<0] = sigma2_sq[g<0]
            g[g<0] = 0
            sv_sq[sv_sq<=eps] = eps
            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num/den

        return vifp