from liveness.quality.metrics.factory import metric_factory
from liveness.quality.metric_vector import DefaultMetricVectorCreator
import cv2
import logging


def main():
    logger = logging.getLogger()

    image = cv2.imread('/home/ryan/datasets/nuaa/ClientRaw/0001/0001_00_00_02_2.jpg')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_image = cv2.GaussianBlur(image, (5, 5), 0)

    metrics_names = [
        "ad",
        "biqi",
        "gme",
        "gpe",
        "hlfi",
        "jqi",
        "lmse",
        "mams",
        "mas",
        "md",
        "mse",
        "nae",
        "niqe",
        "nxc",
        "psnr",
        "ramd",
        # "rred",
        "sc",
        "sme",
        "snr",
        "spe",
        "ssim",
        "tcd",
        "ted",
        "vifd"
    ]
    metrics = metric_factory(metrics_names, logger)
    print(metrics)
    vector_creator = DefaultMetricVectorCreator(metrics)
    vector = vector_creator.create_vector(image, gaussian_image)
    print(vector)


if __name__ == '__main__':
    main()
