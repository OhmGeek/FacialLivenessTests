from liveness.quality.metrics.generic import AbstractQualityMetric
from skimage.measure import compare_ssim as ssim

class StructuralSimilarityMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        return ssim(image, blurred_image)