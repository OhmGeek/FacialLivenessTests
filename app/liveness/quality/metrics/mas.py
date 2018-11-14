""" Mean Angle Similarity """
import numpy as np

from metrics.generic import AbstractQualityMetric

def get_angle_of_images(image, blurred_image):
    scalar_product = np.dot(image, blurred_image)
    
    mag_image = np.sqrt(image.dot(image))
    mag_blurred_image = np.sqrt(blurred_image.dot(blurred_image))

    alpha = (2/np.pi) * np.arccos(scalar_product / (mag_image * mag_blurred_image))

    return alpha

class MeanAngleSimilarityMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):
        """
        Calculate mean angle similarity in the image
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

        # Then, find the mean of the alphas.
        alpha_mean = alphas.mean()

        return 1.0 - alpha_mean
