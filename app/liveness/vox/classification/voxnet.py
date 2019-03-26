"""
This contains the VoxNet model used to classify the 3d structure.
"""

import keras
from keras import layers
from keras.models import Model as keras_Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv3D, MaxPooling3D, Lambda, LeakyReLU, Reshape
from keras import Sequential
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.engine.input_layer import Input
from keras.backend import tf
from liveness.cnn.residual.block import add_common_layers, residual_block
from liveness.generic import AbstractModel
import h5py

class VoxNet(AbstractModel):
    def __init__(self, logger, default_img_dimensions=(224,224), nb_channels=3, cardinality=32):
        self._img_height = default_img_dimensions[1]
        self._img_width = default_img_dimensions[0]
        self._nb_channels = nb_channels
        self._cardinality = cardinality

        self._model = self._create_model()
        self._is_model_created = False

        # Now let's go ahead and create the base object.
        super().__init__(logger)

    def train(self, x, y):
        self._model.fit(x, y, batch_size=32, epochs=2, verbose=1, validation_split=0.33, shuffle=True)

    def test(self, x, y):
        score = self._model.evaluate(x, y, verbose=1)
        return score

    def evaluate(self, image):
        score = self._model.predict(image)
        return score
    
    # -- Generators: both for fitting and testing.
    def fit_generator(self, generator, steps_per_epoch=None, epochs=1, shuffle=True, verbose=1, validation_data=None):
        return self._model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, shuffle=shuffle, verbose=verbose, validation_data=validation_data)

    def test_generator(self, generator):
        score = self._model.evaluate_generator(generator, verbose=1, steps=500)
        return score
    
    # -- Override the base.
    def save(self, pickle_path):
        self._model.save_weights(pickle_path)
    
    def load(self, pickle_path):
        self._model.load_weights(pickle_path)

    # -- Private functions
    def _create_model(self, learning_rate=0.0001):
        model = Sequential()

        model.add(Reshape([1,192,192,200]))
        model.add(Conv3D(32, (5,5,5), strides=(2,2,2)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv3D(32, (3,3,3), strides=(1,1,1)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling3D(strides=(2,2,2)))

        model.add(Flatten())

        # Now the dense classifier
        model.add(Dropout(0.4))
        model.add(Dense(128))
        model.add(Dropout(0.4))
        model.add(Dense(2, activation='sigmoid'))

        model.build(input_shape=(None, 200,192,192))
        model.summary() ## TODO make this be called seperately.

        opt_adam = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy', 'mean_squared_error'])

        self._model = model

        return model