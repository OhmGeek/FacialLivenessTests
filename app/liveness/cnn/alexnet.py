# Alexnet
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.engine.input_layer import Input
import numpy as np
import cv2

def create_model():
    # First, create an empty sequential model.
    model = Sequential()

    # model.add(Input(shape=(None, None, 3)))
    # First, image resize layer (take any dimensional image down to 224 x 224 with 3 channels).
    # we do this by creating a lambda (through the predefined function above)
    # model.add(Lambda(image_resize))
    model.add(Lambda(lambda x: tf.image.resize_images(x,[320,240]), input_shape=(None,None,3)))
    # 1st Conv Layer
    model.add(Conv2D(filters=96, input_shape=(320,240,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('tanh'))

    # Max pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())

    # Now the 2nd convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('tanh'))

    # Max pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(BatchNormalization())

    # 3rd Conv Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('tanh'))

    model.add(BatchNormalization())

    # 4th Conv Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('tanh'))

    model.add(BatchNormalization())

    # 5th Conv Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(BatchNormalization())

    # Now, we pass the conv layer to a dense ff layer.
    model.add(Flatten())

    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(320*240*3,)))
    model.add(Activation('tanh'))

    model.add(Dropout(0.1))

    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(2000))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Dense(1000))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model