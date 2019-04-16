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
from liveness.generic import AbstractModel
import h5py
from keras.models import load_model




class ResidualNetwork(AbstractModel):
    def __init__(self, logger, default_img_dimensions=(224,224), nb_channels=3, cardinality=32, learning_rate=0.001):
        self._img_height = default_img_dimensions[1]
        self._img_width = default_img_dimensions[0]
        self._nb_channels = nb_channels
        self._cardinality = cardinality

        self._model = self._create_model(learning_rate)
        self._is_model_created = False

        # Now let's go ahead and create the base object.
        super().__init__(logger)

    def train(self, x, y):
        self._model.fit(x, y, batch_size=64, epochs=1, verbose=1, validation_split=0.33, shuffle=True)

    def test(self, x, y):
        score = self._model.evaluate(x, y, verbose=1)
        return score

    def evaluate(self, image):
        score = self._model.predict(image)
        return score
    
    # -- Generators: both for fitting and testing.
    def fit_generator(self, generator, steps_per_epoch=None, epochs=1, shuffle=True, verbose=1, validation_data=None, validation_steps=None):
        return self._model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, shuffle=shuffle, verbose=verbose, validation_data=validation_data, validation_steps=validation_steps)

    def test_generator(self, generator):
        score = self._model.evaluate_generator(generator, verbose=1, steps=500)
        return score
    
    # -- Override the base.
    def save(self, pickle_path):
        self._model.save(pickle_path)
    
    def load(self, pickle_path):
        self._model = load_model(pickle_path)

    # -- Private functions
    def _create_model(self, learning_rate=0.001):
        cnn_model = ResNet50(include_top=False, weights='imagenet', input_shape=None)
 
        final_network = Sequential()
        final_network.add(Lambda(lambda img: tf.image.resize_images(img,[self._img_width,self._img_height]), input_shape=(None,None,3)))
        final_network.add(Lambda(lambda img: tf.image.convert_image_dtype(img, tf.float32)))
        final_network.add(cnn_model)
        final_network.add(Flatten())
        final_network.add(Dense(200, use_bias=False))
        final_network.add(BatchNormalization())
        final_network.add(Activation('relu')) 
        final_network.add(Dense(100, use_bias=False))
        final_network.add(BatchNormalization())
        final_network.add(Activation('relu'))
       
        final_network.add(Dropout(0.3))
        final_network.add(Dense(100, use_bias=False))
        final_network.add(BatchNormalization())
        final_network.add(Activation('relu'))
        final_network.add(Dropout(0.3))
       
        final_network.add(Dense(100, use_bias=False))
        final_network.add(BatchNormalization())
        final_network.add(Activation('relu'))
        final_network.add(Dropout(0.3))

        final_network.add(Dense(75, use_bias=False))
        final_network.add(BatchNormalization())
        final_network.add(Activation('relu'))
        final_network.add(Dropout(0.3))

        final_network.add(Dense(50, use_bias=False))
        final_network.add(BatchNormalization())
        final_network.add(Activation('relu'))
        final_network.add(Dropout(0.3))

        final_network.add(Dense(50, use_bias=False))
        final_network.add(BatchNormalization())
        final_network.add(Activation('relu'))
        final_network.add(Dropout(0.3))

        final_network.add(Dense(50, use_bias=False))
        final_network.add(BatchNormalization())
        final_network.add(Activation('relu'))


        final_network.add(Dropout(0.6))
        final_network.add(Dense(1, activation='relu'))
        final_network.add(Activation('sigmoid'))

        # Now freeze all but the base convolutional layer in resnet.
        for layer in cnn_model.layers:
            layer.trainable = False
        
        # Only last two layers are trainable.
        cnn_model.layers[-1].trainable = True
        # set model. 
        self._model = final_network
        self._is_model_created = True

        opt_adam = keras.optimizers.Adam(lr=learning_rate)
        self._model.compile(loss='binary_crossentropy', optimizer=opt_adam, metrics=['accuracy', 'mean_squared_error'])
        self._model.build(input_shape=(None, None, 3))
        self._model.summary() ## TODO make this be called seperately.

        return final_network
