"""
Reduced Ref Entropic Distance
"""
import numpy as np
import pyPyrTools as ppt

from liveness.quality.metrics.generic import AbstractQualityMetric

def ind2wtree(img):
    pass

def extract_red_info(img):
    """ Based on code from file extract_red_info.m """
    subband = 16
    sigma2_w = 0.1
    blk = 3

    # perform wavelet decomposition
    pyr = ppt.Spyr.Spyr(img, height=4, filter='sp5Filters', edges='reflect1')
    print(pyr)
    # imsband = ind2wtree(p)
    # todo not quite sure what to do here.
    #crop to exact multiple size

    # find covariance matrix for U

    # correct possible negative eigenvalues without changing overall variance

    # calculate local variance parameters

    # compute eigenvalues

    # compute entropy, and scale them.

class ReducedRefEntropicDistance(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        dim = None # TODO work out what DIM is.
        red_info_ref = extract_red_info(image)
        red_info_dis = extract_red_info(blurred_image)

        rred = sum(sum(abs(red_info_ref - red_info_dis))) / dim