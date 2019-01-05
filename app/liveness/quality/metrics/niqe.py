from liveness.quality.metrics.generic import AbstractNoReferenceQualityMetric
from pyvideoquality.niqe import niqe
from skimage import img_as_float
class NaturalnessEstimator(AbstractNoReferenceQualityMetric):
    def calculate(self, image):
        img = img_as_float(image)
        return niqe(img)        