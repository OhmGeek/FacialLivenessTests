import logging
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.engine.input_layer import Input
import numpy as np
import cv2
from datasets.nuaa import NUAADataset
from keras.backend import tf


def create_model():
    # First, create an empty sequential model.
    model = Sequential()

    # model.add(Input(shape=(None, None, 3)))
    # First, image resize layer (take any dimensional image down to 224 x 224 with 3 channels).
    # we do this by creating a lambda (through the predefined function above)
    # model.add(Lambda(image_resize))
    model.add(Lambda(lambda x: tf.image.resize_images(x,[224,224]), input_shape=(None,None,3)))
    # 1st Conv Layer
    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))

    # Max pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())

    # Now the 2nd convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    # Max pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(BatchNormalization())

    # 3rd Conv Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    model.add(BatchNormalization())

    # 4th Conv Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    model.add(BatchNormalization())

    # 5th Conv Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(BatchNormalization())

    # Now, we pass the conv layer to a dense ff layer.
    model.add(Flatten())

    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))

    model.add(Dropout(0.4))

    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


def main():
    # First, fetch the two distinct sets of data.
    # Two neurons: [1.0, 0.0] -> fake, [0.0, 1.0] -> real
    # For each image in X, resize to (224,224) with 3 channels. Use OpenCV.

    # Now create the CNN model
    model = create_model()


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build()
    model.summary()

    # Now create the training set.
    dataset = NUAADataset(logging.getLogger("c.o.datasets.nuaa"), "/home/ryan/datasets/nuaa/")
    dataset.pre_process()

    imposter_set = dataset.read_dataset("ImposterRaw")
    imposter_y = np.tile([1.0, 0.0], (imposter_set.shape[0], 1))

    client_set = dataset.read_dataset("ClientRaw")
    client_y = np.tile([0.0, 1.0], (client_set.shape[0], 1))

    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    # Train the model on our training set.
    model.fit(x, y, batch_size=64, epochs=5, verbose=1, validation_split=0.7, shuffle=True)

    dataset.close() # Important, to close the file.

if __name__ == "__main__":
    main()