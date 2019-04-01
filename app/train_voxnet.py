import logging
import keras
from liveness.vox.classification.voxnet import VoxNet
from liveness.vox.reconstruction.vrn import FaceVoxelBuilder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from keras.layers.normalization import BatchNormalization
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from image_data_generator import DataGenerator
from keras.optimizers import Adam
from keras.engine.input_layer import Input
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import numpy as np
import cv2
from datasets.mad import MaskAttackDataset
from datasets.nuaa import NUAADataset
from datasets.replayattack import ReplayAttackDataset
import face_recognition
from PIL import Image
import sys

def main():
    # First, fetch the two distinct sets of data.
    # Two neurons: [1.0, 0.0] -> fake, [0.0, 1.0] -> real
    # For each image in X, resize to (224,224) with 3 channels. Use OpenCV.
    sys.setrecursionlimit(10000)

    print(sys.getrecursionlimit())
    # Now create the CNN model
    model = VoxNet(logging.Logger("voxnet"), learning_rate=0.1)
    
    # adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
    # model.summary()

    # Now create the training set.
    dataset = MaskAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ohmgeek_default/datasets/mad/", subjects=[1])
    #dataset = NUAADataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ohmgeek_default/datasets/nuaa")
    dataset.pre_process()

    imposter_set = dataset.read_dataset("C")
    imposter_y = np.tile([0.0], imposter_set.shape[0])

    client_set = dataset.read_dataset("B")
    client_y = np.tile([1.0], client_set.shape[0])

    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    x,y = shuffle(x, y)

    # Train the model on our training set.
    batch_size = 4
    generator = DataGenerator(x, y, batch_size=batch_size)

    steps_per_epoch = len(x) / batch_size
    model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=5, shuffle=True, verbose=1)
    # model.save('alexnet.h5')

    dataset = None
    x = None
    y = None
    imposter_set = None
    imposter_y = None
    client_set = None
    client_y = None

    # Now create the training set.
    dataset = MaskAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ohmgeek_default/datasets/mad/", subjects=[8,9,10,11,12,13,14,15,16,17])
    dataset.pre_process()

    imposter_set = dataset.read_dataset("C")
    imposter_y = np.tile([0.0], imposter_set.shape[0])

    client_set = dataset.read_dataset("B")
    client_y = np.tile([1.0], client_set.shape[0])

    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    x,y = shuffle(x, y)
    
    generator = gen.flow(x, y, batch_size=8)
    score = model.test_generator(generator)
    print("Final Accuracy is: " + str(score))
    #model.save('alexnet.h5')
    dataset.close() # Important, to close the file.
    
if __name__ == "__main__":
    main()
