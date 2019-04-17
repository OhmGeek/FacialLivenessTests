from datasets.replayattack import ReplayAttackDataset
from liveness.generic import DummyLivenessTest
from liveness.cnn.residual.model import ResidualNetwork
from testframework.tests import TestDummyCase
from testframework.runner import TestRunner
import cv2
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.backend import manual_variable_initialization
from train_cnn import pre_process_fn
def main():
    manual_variable_initialization(True)
    # first, set log level to display everything we want
    # TODO: change this to warn for production.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    print("Running test.py")
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ohmgeek_default/datasets/replay-attack/", mode='test')
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")[:10]
    output_imposter = [0.0 for x in range(imposter_set.shape[0])]

    client_set = dataset.read_dataset("real")[:10]
    output_client = [1.0 for x in range(client_set.shape[0])]
    # Load the model.
    model = ResidualNetwork(logger)
    model.load('/home/ryan/Downloads/alexnet.h5')

    # Merge the data together.
    input_x = np.concatenate((imposter_set, client_set))
    input_y = np.concatenate((output_imposter, output_client))

    score = model.test(input_x, input_y)

   
    print("Total results")
    print(score)

    preprocess_fn_numpy = np.vectorize(pre_process_fn)
    y_pred = model.evaluate(pre_process_fn(x))
    print(y_pred)
    tn, fp, fn, tp = confusion_matrix(input_y, y_pred).ravel()

    print("True Negatives: ", tn)
    print("False Positives: ", fp)
    print("False Negatives: ", fn)
    print("True Positives: ", tp)

if __name__ == "__main__":
    main()
