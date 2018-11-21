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
<<<<<<< HEAD
from metrics.gme import  GradientMagnitudeErrorMetric
from metrics.gpe import  GradientPhaseErrorMetric
from metrics.sme import  SpectralMagnitudeErrorMetric
from metrics.spe import  SpectralPhaseErrorMetric
from metrics.tcd import  TotalCornerDifferenceMetric
=======
from metrics.gme import GradientMagnitudeErrorMetric
from metrics.gpe import GradientPhaseErrorMetric
from metrics.sme import SpectralMagnitudeErrorMetric
from metrics.spe import SpectralPhaseErrorMetric
from metrics.tcd import TotalCornerDifferenceMetric
from metrics.mas import MeanAngleSimilarityMetric
from metrics.mams import MeanAngleMagnitudeSimilarityMetric
>>>>>>> 5078193cbffb79472a0b08a5132782fe4cc54c48
import cv2
import logging
def main():
    logger = logging.getLogger()
<<<<<<< HEAD
    metrics = [MaximumDifferenceMetric(logger),MeanSquaredErrorMetric(logger),NormalisedAbsoluteErrorMetric(logger),NormalisedCrossCorrelationMetric(logger),PeakSignalToNoiseRatioMetric(logger),StructuralContentMetric(logger),SignalToNoiseRatioMetric(logger),AverageDifferenceMetric(logger),LaplacianMeanSquaredMetric(logger),GradientMagnitudeErrorMetric(logger),GradientPhaseErrorMetric(logger),SpectralMagnitudeErrorMetric(logger),SpectralPhaseErrorMetric(logger), TotalCornerDifferenceMetric(logger)]
    image = cv2.imread('/home/ryan/datasets/nuaa/ClientRaw/0001/0001_00_00_02_2.jpg')
=======
    metrics = [MeanAngleMagnitudeSimilarityMetric(logger)]
    image = cv2.imread('/home/ryan/datasets/nuaa/ClientRaw/0001/0001_00_00_01_2.jpg')
>>>>>>> 5078193cbffb79472a0b08a5132782fe4cc54c48
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_image = cv2.GaussianBlur(image,(5,5),0)
    vector_factory = DefaultMetricVectorCreator(metrics)

    output_vector = vector_factory.create_vector(image, gaussian_image)
    print(output_vector)

if __name__ == '__main__':
    main()