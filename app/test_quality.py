from datasets.replayattack import ReplayAttackDataset
from liveness.generic import DummyLivenessTest
from liveness.quality.model import QualityLDAModel
from testframework.tests import TestDummyCase
from testframework.runner import TestRunner
from liveness.quality.metrics.factory import metric_factory
from liveness.quality.metric_vector import DefaultMetricVectorCreator
import cv2
import logging
import numpy as np
from sklearn.metrics import confusion_matrix

def main():
    # first, set log level to display everything we want
    # TODO: change this to warn for production.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    print("Running test.py")
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ryan/datasets/replay-attack/", mode='devel')
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")[:10]
    output_imposter = [0.0 for x in range(imposter_set.shape[0])]

    client_set = dataset.read_dataset("real")[:10]
    output_client = [1.0 for x in range(client_set.shape[0])]
    # Load the model.
    model = QualityLDAModel(logger)
    model.load('/home/ryan/Documents/dev/LivenessTests/models/lda_model_v2.pkl')

    # Merge the data together.
    input_x = np.concatenate((imposter_set, client_set))
    input_y = np.concatenate((output_imposter, output_client))

    score = model.test(input_x, input_y)

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
