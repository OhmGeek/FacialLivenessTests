from datasets.nuaa import NUAADataset
from datasets.replayattack import ReplayAttackDataset
from datasets.mad import MaskAttackDataset
from liveness.generic import DummyLivenessTest
from liveness.quality.model import QualitySVMModel, QualityLDAModel
from testframework.tests import TestDummyCase
from testframework.runner import TestRunner
from liveness.quality.metrics.factory import metric_factory
from liveness.quality.metric_vector import DefaultMetricVectorCreator
import cv2
import logging
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

def main():
    # first, set log level to display everything we want
    # TODO: change this to warn for production.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
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
    vector_creator = DefaultMetricVectorCreator(metrics)

    print("Running test.py")
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattackdevel"), "/home/ryan/datasets/replayAttackDB/")
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")[:100] # ImposterRaw
    client_set = dataset.read_dataset("real")[:100] # ClientRaw
    # Divide dataset into train, and test (40%, 60%)
    train_vectors = []
    train_outputs = []
    for imposter_img in imposter_set[: int(imposter_set.shape[0])]:
        train_vectors.append(imposter_img)
        train_outputs.append(0.0) # 0.0 -> fake

    for client_img in client_set[: int(client_set.shape[0])]:
        train_vectors.append(client_img)
        train_outputs.append(1.0) # 1.0 -> real
    
    train_vectors, train_outputs = shuffle(train_vectors, train_outputs)
    model = QualityLDAModel(logging.Logger("lda_model"))
    # Evaluate on testing set
    print("Now training")
    print(len(train_vectors), len(train_outputs))
    model.train(train_vectors, train_outputs)
    print("Trained.")
    print("")
    print("Now saving")
    model.save('/home/ryan/Documents/dev/LivenessTests/models/lda_model_v2.pkl')
    print("Saved.")

    # Now we train with ReplayAttack
    print("Now load replay attack for testing")
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattacktest"), "/home/ryan/datasets/replayAttackDB/", mode='test')
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")[:100]
    output_imposter = [0.0 for x in range(imposter_set.shape[0])]

    client_set = dataset.read_dataset("real")[:100]
    output_client = [1.0 for x in range(client_set.shape[0])]
    
    # Merge the data together.
    input_x = np.concatenate((imposter_set, client_set))
    input_y = np.concatenate((output_imposter, output_client))

    input_x, input_y = shuffle(input_x, input_y)

    print("Get ready to test")
    score = model.test(input_x, input_y)
    print("testing finished")
    y_pred = model.evaluate(input_x)
    tn, fp, fn, tp = confusion_matrix(input_y, y_pred).ravel()

    print("Total results")
    print(score)

    print("True Negatives: ", tn)
    print("False Positives: ", fp)
    print("False Negatives: ", fn)
    print("True Positives: ", tp)

if __name__ == "__main__":
    main()
