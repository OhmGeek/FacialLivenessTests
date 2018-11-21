""" High Low Frequency index Index """

from liveness.quality.metrics.generic import AbstractNoReferenceQualityMetric
import numpy as np
import math
class HighLowFrequencyIndexMetric(AbstractNoReferenceQualityMetric):
    def calculate(self, image):
        raise NotImplementedError()


