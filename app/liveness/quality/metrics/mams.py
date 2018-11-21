""" Mean Angle Magnitude Similarity """
import numpy as np

from liveness.quality.metrics.generic import AbstractQualityMetric

class MeanAngleMagnitudeSimilarityMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate mean angle magnitude similarity in the image
        :param image: full quality image
        :param blurred_image: gaussian blurred quality image
        :return: the mean angle similarity
        """

        scalar_products = np.multiply(image, blurred_image)

        # Find the magnitudes.
        magnitude_product = np.multiply(np.abs(image), np.abs(blurred_image)) 

        # Deal with the error case where the magnitudes are 0 (because they might sometimes be)
        magnitude_product[magnitude_product == 0] = 1

        # Calculate the alpha values for the entire image.
        alphas = (2 / np.pi) * np.arccos(np.divide(scalar_products, magnitude_product))

        # Then, subtract 1 from this angle.
        ones = np.ones_like(alphas)
        alphas = ones - alphas

        magnitude_calc = np.abs(np.subtract(image, blurred_image)) / 255
        magnitude_calc = np.subtract(ones, magnitude_calc)

        output = np.subtract(ones, np.multiply(alphas, magnitude_calc))

        return output.mean()