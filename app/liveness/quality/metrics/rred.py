"""
Reduced Ref Entropic Distance
"""
import numpy as np

from liveness.quality.metrics.generic import AbstractQualityMetric


class ReducedRefEntropicDistance(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        