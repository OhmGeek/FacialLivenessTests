from metric_vector import DefaultMetricVectorCreator
from metrics.md import MaximumDifferenceMetric
from metrics.mse import MeanSquaredErrorMetric
from metrics.nae import NormalisedAbsoluteErrorMetric
from metrics.nxc import NormalisedCrossCorrelationMetric
from metrics.psnr import PeakSignalToNoiseRatioMetric
from metrics.sc import StructuralContentMetric
from metrics.snr import SignalToNoiseRatioMetric
from metrics.ad import AverageDifferenceMetric
from metrics.lmse import LaplacianMeanSquaredMetric
from metrics.gme import GradientMagnitudeErrorMetric
from metrics.gpe import GradientPhaseErrorMetric
from metrics.sme import SpectralMagnitudeErrorMetric
from metrics.spe import SpectralPhaseErrorMetric
from metrics.tcd import TotalCornerDifferenceMetric
import cv2
import logging
def main():
    logger = logging.getLogger()
    metrics = [TotalCornerDifferenceMetric(logger)]
    image = cv2.imread('/home/ryan/datasets/nuaa/ClientRaw/0001/0001_00_00_01_2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_image = cv2.GaussianBlur(image,(5,5),0)
    vector_factory = DefaultMetricVectorCreator(metrics)

    output_vector = vector_factory.create_vector(image, gaussian_image)
    print(output_vector)

if __name__ == '__main__':
    main()