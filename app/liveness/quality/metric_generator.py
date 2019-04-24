import cv2

from liveness.quality.metric_vector import DefaultMetricVectorCreator
from liveness.quality.metrics.factory import metric_factory


def preprocessor(data, outputs, logger):
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
    vector_creator = DefaultMetricVectorCreator(metrics)
    train_vectors = []
    train_outputs = []
    for i in range(len(data)):
        try:
            client_img = data[i]
            image = cv2.cvtColor(client_img, cv2.COLOR_BGR2GRAY)

            gaussian_image = cv2.GaussianBlur(image, (5, 5), 0)
            vector = vector_creator.create_vector(image, gaussian_image)
            # None can't be in the vector. Everything must be a number.
            if None in vector:
                raise Exception()

            train_vectors.append(vector)

            if outputs is not None:
                train_outputs.append(outputs[i])
        except Exception as e:
            logger.error("Error while evaluating image")
            print(e)
            raise e
    return train_vectors, train_outputs
