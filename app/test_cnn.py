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
from train_cnn import pre_process_fn, preprocess_fn_all
def main():
    manual_variable_initialization(True)
    # first, set log level to display everything we want
    # TODO: change this to warn for production.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    print("Running test.py")
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ohmgeek_default/datasets/replay-attack/", mode='test')
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")
    output_imposter = [[1.0, 0.0] for x in range(imposter_set.shape[0])]

    client_set = dataset.read_dataset("real")
    output_client = [[0.0, 1.0] for x in range(client_set.shape[0])]
    # Load the model.
    model = ResidualNetwork(logger)
    model.load('/home/ohmgeek_default/LivenessTests/app/alexnet.h5')

    # Merge the data together.
    input_x = np.concatenate((imposter_set, client_set))
    input_y = np.concatenate((output_imposter, output_client))

    score = model.test(preprocess_fn_all(input_x), input_y)
    print("Accuracy:") 
    print(score)

    y_pred = model.evaluate(preprocess_fn_all(input_x))
    
    thresh = 0.6
    print("Average:", np.average(y_pred))
    print("Min:", np.amin(y_pred))
    print("Max:", np.amax(y_pred))
    y_pred = y_pred.argmax(axis=-1)

    #y_pred[y_pred > thresh] = 1
    #y_pred[y_pred <= thresh] = 0

    print(y_pred)
    c_matrix = confusion_matrix(input_y, y_pred)

    print("Confusion Matrix:")
    print(c_matrix)
    dataset.close() # Important, to close the file.

if __name__ == "__main__":
    main()
