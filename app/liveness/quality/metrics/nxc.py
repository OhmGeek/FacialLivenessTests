"""
Average Difference Quality Metric

Simply the average difference I - G for all the pixels
where I is the original image, and G is the gaussian blurred image.

"""
import numpy as np

from liveness.quality.metrics.generic import AbstractQualityMetric


class NormalisedCrossCorrelationMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        # First, find image dotted with blurred_image
        dot_product = np.dot(image, np.transpose(blurred_image))

        # Then square the components of image
        image_squared = np.square(image)

        # Then do the division and return the result.
        output = dot_product.sum() / image_squared.sum()

        return output
