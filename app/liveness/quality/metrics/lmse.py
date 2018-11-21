"""
Laplacian Mean Squared Error metric.

"""
from liveness.quality.metrics.generic import AbstractQualityMetric
import numpy as np
import math

def laplacian(x, i, j):
    if(i < 0 or j < 0 or i >= x.shape[0] - 1 or j >= x.shape[1] - 1):
        return 0.0
    # http://www.irdindia.in/journal_ijaeee/pdf/vol2_iss6/3.pdf is the source
    # of how to implement this.

    return math.ceil((x[i+1, j] + x[i-1, j] + x[i, j+1] + x[i, j-1] - (4*x[i,j])))

def laplacian_of_image(image):
    # TODO: make this faster. This is a bottleneck.
    output = np.zeros_like(image)
    for row_index, row in enumerate(image):
        for col_index, col in enumerate(row):
            val = laplacian(image, row_index, col_index)
            output[row_index][col_index] = val
    return output

class LaplacianMeanSquaredMetric(AbstractQualityMetric):
    def calculate(self, image, blurred_image):

        laplacian_image = laplacian_of_image(image)
        laplacian_blurred_image = laplacian_of_image(blurred_image)

        numerator = np.subtract(laplacian_image, laplacian_blurred_image) # Difference
        numerator = np.square(numerator) # Square the difference
        numerator = numerator.sum() # Sum the squared differences.

        denominator = np.square(laplacian_image)
        denominator = denominator.sum() # sum the squared value.

        return (numerator / denominator)