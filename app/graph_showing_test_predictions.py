from datasets.replayattack import ReplayAttackDataset
from liveness.generic import DummyLivenessTest
from liveness.cnn.residual.model import ResidualNetwork
from testframework.tests import TestDummyCase
from testframework.runner import TestRunner
from liveness.quality.model import QualityLDAModel
from liveness.quality.metric_vector import DefaultMetricVectorCreator
import cv2
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.backend import manual_variable_initialization
from train_cnn import pre_process_fn, preprocess_fn_all
import matplotlib.pyplot as plt

def main():
    manual_variable_initialization(True)
    # first, set log level to display everything we want
    # TODO: change this to warn for production.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ryan/datasets/replay-attack/", mode='test')
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")[:100]

    client_set = dataset.read_dataset("real")[:100]
    # Load the model.
    model = ResidualNetwork(logger)
    model.load('/home/ryan/Documents/dev/LivenessTests/models/cnn_v3.h5')

    # Merge the data together.
    input_x = np.concatenate((imposter_set, client_set))

    y_pred_resnet = model.evaluate(preprocess_fn_all(input_x))
    y_pred_resnet = y_pred_resnet[:,1]
    ## Now do the same with the Quality Metric.
    model = None
    model = QualityLDAModel(logger)
    model.load('/home/ryan/Documents/dev/LivenessTests/models/lda_model_v2.pkl')
    y_pred_quality = model.evaluate(input_x, get_probability=True)
    y_pred_quality = y_pred_quality[:, 1]

    fig, ax = plt.subplots()
    
    ax.scatter(y_pred_quality[:100], y_pred_resnet[:100], color='red', label='Fake')
    ax.scatter(y_pred_quality[100:], y_pred_resnet[100:], color='green', label='Real')
    
    ax.legend()
    
    plt.ylabel('ResNet Prediction Output (0/1)')
    plt.xlabel("Quality Prediction Output (0/1)")
    plt.title("Output predictions per image, with the two metrics")
    plt.show()

    dataset.close() # Important, to close the file.


if __name__ == "__main__":
    main()
