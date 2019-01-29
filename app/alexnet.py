import logging
import keras
from liveness.cnn.residual.model import ResidualNetwork
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.engine.input_layer import Input
import numpy as np
import cv2
from datasets.nuaa import NUAADataset
from datasets.replayattack import ReplayAttackDataset

def main():
    # First, fetch the two distinct sets of data.
    # Two neurons: [1.0, 0.0] -> fake, [0.0, 1.0] -> real
    # For each image in X, resize to (224,224) with 3 channels. Use OpenCV.

    # Now create the CNN model
    model = ResidualNetwork(logging.Logger("resnet"))
    # adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
    # model.summary()

    # Now create the training set.
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ryan/datasets/replayAttackDB/")
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")
    imposter_y = np.tile([1.0, 0.0], (imposter_set.shape[0], 1))

    client_set = dataset.read_dataset("real")
    client_y = np.tile([0.0, 1.0], (client_set.shape[0], 1))

    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    # Train the model on our training set.
    model.train(x, y)

    dataset = None
    # Now create the training set.
    dataset = NUAADataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ryan/datasets/nuaa/")
    dataset.pre_process()

    imposter_set = dataset.read_dataset("ImposterRaw")
    imposter_y = np.tile([1.0, 0.0], (imposter_set.shape[0], 1))

    client_set = dataset.read_dataset("ClientRaw")
    client_y = np.tile([0.0, 1.0], (client_set.shape[0], 1))

    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    score = model.test(x, y)
    print("Final Accuracy is: " + str(score[1]))
    model.save('alexnet.h5')
    dataset.close() # Important, to close the file.

if __name__ == "__main__":
    main()