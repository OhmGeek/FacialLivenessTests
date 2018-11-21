from metric_vector import DefaultMetricVectorCreator
from metrics.md import   MaximumDifferenceMetric
from metrics.mse import  MeanSquaredErrorMetric
from metrics.nae import  NormalisedAbsoluteErrorMetric
from metrics.nxc import  NormalisedCrossCorrelationMetric
from metrics.psnr import PeakSignalToNoiseRatioMetric
from metrics.sc import   StructuralContentMetric
from metrics.snr import  SignalToNoiseRatioMetric
from metrics.ad import   AverageDifferenceMetric
from metrics.lmse import LaplacianMeanSquaredMetric
from metrics.gme import  GradientMagnitudeErrorMetric
from metrics.gpe import  GradientPhaseErrorMetric
from metrics.sme import  SpectralMagnitudeErrorMetric
from metrics.spe import  SpectralPhaseErrorMetric
from metrics.tcd import  TotalCornerDifferenceMetric
from metrics.gme import GradientMagnitudeErrorMetric
from metrics.gpe import GradientPhaseErrorMetric
from metrics.sme import SpectralMagnitudeErrorMetric
from metrics.spe import SpectralPhaseErrorMetric
from metrics.tcd import TotalCornerDifferenceMetric
from metrics.mas import MeanAngleSimilarityMetric
from metrics.mams import MeanAngleMagnitudeSimilarityMetric
from metrics.jqi import JPEGQualityIndexMetric
import cv2
import logging
def main():
    logger = logging.getLogger()

    image = cv2.imread('/home/ryan/datasets/nuaa/ClientRaw/0001/0001_00_00_02_2.jpg')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_image = cv2.GaussianBlur(image,(5,5),0)

    m = JPEGQualityIndexMetric(logger)
    print(m.calculate(image))


if __name__ == '__main__':
    main()