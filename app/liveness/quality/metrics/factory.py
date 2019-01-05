from liveness.quality.metrics.ad import AverageDifferenceMetric
from liveness.quality.metrics.biqi import BlindImageQualityIndex
from liveness.quality.metrics.gme import GradientMagnitudeErrorMetric
from liveness.quality.metrics.gpe import GradientPhaseErrorMetric
from liveness.quality.metrics.hlfi import HighLowFrequencyIndexMetric
from liveness.quality.metrics.jqi import JPEGQualityIndexMetric
from liveness.quality.metrics.lmse import LaplacianMeanSquaredMetric
from liveness.quality.metrics.mams import MeanAngleMagnitudeSimilarityMetric
from liveness.quality.metrics.mas import MeanAngleSimilarityMetric
from liveness.quality.metrics.md import MaximumDifferenceMetric
from liveness.quality.metrics.mse import MeanSquaredErrorMetric
from liveness.quality.metrics.nae import NormalisedAbsoluteErrorMetric
from liveness.quality.metrics.niqe import NaturalnessEstimator
from liveness.quality.metrics.nxc import NormalisedCrossCorrelationMetric
from liveness.quality.metrics.psnr import PeakSignalToNoiseRatioMetric
from liveness.quality.metrics.ramd import RAveragedMetric
# from liveness.quality.metrics.rred import ReducedRefEntropicDistance
from liveness.quality.metrics.sc import StructuralContentMetric
from liveness.quality.metrics.sme import SpectralMagnitudeErrorMetric
from liveness.quality.metrics.snr import SignalToNoiseRatioMetric
from liveness.quality.metrics.spe import SpectralPhaseErrorMetric
from liveness.quality.metrics.ssim import StructuralSimilarityMetric
from liveness.quality.metrics.tcd import TotalCornerDifferenceMetric
from liveness.quality.metrics.ted import TotalEdgeDifferenceMetric
from liveness.quality.metrics.vifd import VisualInformationFidelityMetric

def get_instance(name):
    name = name.lower()

    lookup_table = {
        "ad": AverageDifferenceMetric,
        "biqi": BlindImageQualityIndex,
        "gme": GradientMagnitudeErrorMetric,
        "gpe": GradientPhaseErrorMetric,
        "hlfi": HighLowFrequencyIndexMetric,
        "jqi": JPEGQualityIndexMetric,
        "lmse": LaplacianMeanSquaredMetric,
        "mams": MeanAngleMagnitudeSimilarityMetric,
        "mas": MeanAngleSimilarityMetric,
        "md": MaximumDifferenceMetric,
        "mse": MeanSquaredErrorMetric,
        "nae": NormalisedAbsoluteErrorMetric,
        "niqe": NaturalnessEstimator,
        "nxc": NormalisedCrossCorrelationMetric,
        "psnr": PeakSignalToNoiseRatioMetric,
        "ramd": RAveragedMetric,
        # "rred": ReducedRefEntropicDistance,
        "sc": StructuralContentMetric,
        "sme": SpectralMagnitudeErrorMetric,
        "snr": SignalToNoiseRatioMetric,
        "spe": SpectralPhaseErrorMetric,
        "ssim": StructuralSimilarityMetric,
        "tcd": TotalCornerDifferenceMetric,
        "ted": TotalEdgeDifferenceMetric,
        "vifd": VisualInformationFidelityMetric
    }
    return lookup_table.get(name)


def metric_factory(metric_name, logger):
    if type(metric_name) is str:
        return get_instance(metric_name)
    else:
        return [get_instance(m)(logger) for m in metric_name]
