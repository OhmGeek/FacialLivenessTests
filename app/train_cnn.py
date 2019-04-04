import logging
import keras
from liveness.cnn.residual.model import ResidualNetwork
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.optimizers import Adam
from keras.engine.input_layer import Input
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import numpy as np
import cv2
from datasets.nuaa import NUAADataset
from datasets.replayattack import ReplayAttackDataset
from datasets.mad import MaskAttackDataset
import face_recognition
from PIL import Image

def get_largest_bounding_box(locations):
    if len(locations) == 0:
        return None
    w = max(locations, key=lambda x: np.linalg.norm(x[0]-x[2]) * np.linalg.norm(x[1]-x[3]))
    return w

def pre_process_fn(image_arr):
    original_shape = image_arr.shape
    image_arr = image_arr.astype(np.uint8)
    locations = face_recognition.face_locations(image_arr, number_of_times_to_upsample=0, model='cnn')
    
    max_loc = get_largest_bounding_box(locations)
    # If there's an error, just use the whole image.
    if max_loc is None:
        return image_arr

    # Otherwise, isolate the face.
    top, right, bottom, left = max_loc

    dist = max(abs(bottom - top), abs(right - left))

    new_bottom = top + dist
    new_right = left + dist
    face_image = image_arr[top:new_bottom, left:new_right]
    
    # Now, to fix a bug in Keras, resize this image.
    face_image = cv2.resize(face_image, dsize=(original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)

    return (face_image)

def main():
    # First, fetch the two distinct sets of data.
    # Two neurons: [1.0, 0.0] -> fake, [0.0, 1.0] -> real
    # For each image in X, resize to (224,224) with 3 channels. Use OpenCV.

    # Now create the CNN model
    model = ResidualNetwork(logging.Logger("resnet"), learning_rate=0.0001, default_img_dimensions=(224,224))
  
    # adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
    # model.summary()

    # Now create the training set.
    dataset = NUAADataset(logging.getLogger("c.o.datasets.nuaa"), "/home/ohmgeek_default/datasets/nuaa")
    dataset.pre_process()

    imposter_set = dataset.read_dataset("ImposterRaw")
    imposter_y = np.tile([0.0], imposter_set.shape[0])

    client_set = dataset.read_dataset("ClientRaw")
    client_y = np.tile([1.0], client_set.shape[0])

    gen = ImageDataGenerator(horizontal_flip = False,
                         vertical_flip = False,
                         preprocessing_function=pre_process_fn
                        )


    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    x,y = shuffle(x, y)

    # Train the model on our training set.
    batch_size = 64
    generator = gen.flow(x, y, batch_size=batch_size)

    size_of_dataset = len(x)

    dataset = None
    x = None
    y = None
    imposter_set = None
    imposter_y = None
    client_set = None
    client_y = None
    # Now create the training set.
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ryan/datasets/replayAttackDB/")
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")
    imposter_y = np.tile([0.0], imposter_set.shape[0])

    client_set = dataset.read_dataset("real")
    client_y = np.tile([1.0], client_set.shape[0])

    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    x,y = shuffle(x, y)
    
    test_generator = gen.flow(x, y, batch_size=8)



    model.fit_generator(generator, steps_per_epoch=size_of_dataset/batch_size, epochs=15, shuffle=True, verbose=1, validation_data=test_generator)
    model.save('alexnet.h5')

    
    score = model.test_generator(generator)
    print("Final Accuracy is: " + str(score))
    #model.save('alexnet.h5')
    dataset.close() # Important, to close the file.
    
if __name__ == "__main__":
    main()
