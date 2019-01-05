from liveness.quality.metrics.generic import AbstractNoReferenceQualityMetric
import numpy as np
from liveness.quality.helpers.biqi import biqi

class BlindImageQualityIndex(AbstractNoReferenceQualityMetric):
    def calculate(self, image):
        return biqi(image)