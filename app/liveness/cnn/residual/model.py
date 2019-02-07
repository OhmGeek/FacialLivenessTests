import keras
from keras import layers
from keras.models import Model as keras_Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from keras import Sequential
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.engine.input_layer import Input
from keras.backend import tf
from liveness.cnn.residual.block import add_common_layers, residual_block
import h5py

class ResidualNetwork(object):
    def __init__(self, logger, default_img_dimensions=(224,224), nb_channels=3, cardinality=32):
        self._logger = logger
        self._img_height = default_img_dimensions[1]
        self._img_width = default_img_dimensions[0]
        self._nb_channels = nb_channels
        self._cardinality = cardinality

        self._model = self._create_model()
        self._is_model_created = False

    def train(self, x, y):
        self._model.fit(x, y, batch_size=64, epochs=1, verbose=1, validation_split=0.33, shuffle=True)

    def fit_generator(self, generator, steps_per_epoch=None, epochs=1, shuffle=True, verbose=1, validation_data=None):
        return self._model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, shuffle=shuffle, verbose=verbose, validation_data=validation_data)

    def test(self, x, y):
        score = self._model.evaluate(x, y, verbose=1)
        return score

    def test_generator(self, generator):
        score = self._model.evaluate_generator(generator, verbose=1, steps=500)
        return score
    def save(self, pickle_path):
        self._model.save_weights(pickle_path)
    
    def _create_model(self):
        cnn_model = ResNet50(include_top=False, weights='imagenet', input_shape=None)
 
        final_network = Sequential()
        final_network.add(Lambda(lambda img: tf.image.resize_images(img,[self._img_width,self._img_height]), input_shape=(None,None,3)))
        final_network.add(Lambda(lambda img: tf.image.convert_image_dtype(img, tf.float32)))
        final_network.add(cnn_model)
        final_network.add(Flatten())
        final_network.add(Dense(200, activation='relu'))
        final_network.add(Dense(100, activation='relu'))
        final_network.add(Dense(100, activation='relu'))
        final_network.add(Dense(50, activation='relu'))
        final_network.add(Dense(500, activation='relu'))
        final_network.add(Dropout(0.6))
        final_network.add(Dense(2, activation='relu'))
        final_network.add(Activation('softmax'))

        # Now freeze all but the base convolutional layer in resnet.
        for layer in cnn_model.layers:
            layer.trainable = False
        
        # Only last layer is trainable.
        cnn_model.layers[-1].trainable = True

        # set model. 
        self._model = final_network
        self._is_model_created = True

        opt_adam = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)
        self._model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy', 'mean_squared_error'])
        self._model.build(input_shape=(None, None, 3))
        self._model.summary() ## TODO make this be called seperately.

        return final_network
