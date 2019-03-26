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
from datasets.nuaa import NUAADataset
from datasets.replayattack import ReplayAttackDataset
import face_recognition
from PIL import Image
import sys

def get_largest_bounding_box(locations):
    if len(locations) == 0:
        return None
    w = max(locations, key=lambda x: np.linalg.norm(x[0]-x[2]) * np.linalg.norm(x[1]-x[3]))
    return w

def pre_process_fn(image_arr):
    
    face_image = image_arr
    voxel_builder = FaceVoxelBuilder(logging.Logger("VoxelBuilder"))
    face_3d = voxel_builder.build_3d(face_image)

    return (face_3d)

def main():
    # First, fetch the two distinct sets of data.
    # Two neurons: [1.0, 0.0] -> fake, [0.0, 1.0] -> real
    # For each image in X, resize to (224,224) with 3 channels. Use OpenCV.
    sys.setrecursionlimit(10000)

    print(sys.getrecursionlimit())
    # Now create the CNN model
    model = VoxNet(logging.Logger("voxnet"))
    
    # adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
    # model.summary()

    # Now create the training set.
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ohmgeek_default/datasets/replayAttackDB/")
    # dataset = NUAADataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ohmgeek_default/datasets/nuaa")
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")
    imposter_y = np.tile([1.0, 0.0], (imposter_set.shape[0], 1))

    client_set = dataset.read_dataset("real")
    client_y = np.tile([0.0, 1.0], (client_set.shape[0], 1))

    # gen = VoxelDataGenerator(
    #                      preprocessing_function=pre_process_fn,
    #                     )


    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    x,y = shuffle(x, y)
    # folds = list(KFold(n_splits=k, shuffle=True, random_state=1).split(x, y))

    # Train the model on our training set.
    batch_size = 100
    generator = DataGenerator(x, y, batch_size=32)

    steps_per_epoch = len(x) / batch_size
    steps_per_epoch = 1
    model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=5, shuffle=True, verbose=1)
    # model.save('alexnet.h5')

    exit()
    dataset = None
    x = None
    y = None
    imposter_set = None
    imposter_y = None
    client_set = None
    client_y = None

    # Now create the training set.
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ohmgeek_default/datasets/replayAttackDB/")
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")
    imposter_y = np.tile([1.0, 0.0], (imposter_set.shape[0], 1))

    client_set = dataset.read_dataset("real")
    client_y = np.tile([0.0, 1.0], (client_set.shape[0], 1))

    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    x,y = shuffle(x, y)
    
    generator = gen.flow(x, y, batch_size=100)
    score = model.test_generator(generator)
    print("Final Accuracy is: " + str(score))
    #model.save('alexnet.h5')
    dataset.close() # Important, to close the file.
    
if __name__ == "__main__":
    main()
