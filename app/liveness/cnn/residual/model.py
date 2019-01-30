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
        self._model.fit(x, y, batch_size=8, epochs=1, verbose=1, validation_split=0.33, shuffle=True)

    def test(self, x, y):
        score = self._model.evaluate(x, y, verbose=1)
        return score
    
    def _create_model(self):

        # # Now create input
        # y = layers.Input(shape=(None, None, self._nb_channels))
        # x = layers.Lambda(lambda img: tf.image.resize_images(img,[self._img_width,self._img_height]), input_shape=(None,None,3))(y)
        # x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
        # x = add_common_layers(x)

        # # conv2
        # x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # for i in range(3):
        #     project_shortcut = True if i == 0 else False
        #     x = residual_block(x, 128, 256, self._cardinality, _project_shortcut=project_shortcut)

        # # conv3
        # for i in range(4):
        #     # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        #     strides = (2, 2) if i == 0 else (1, 1)
        #     x = residual_block(x, 256, 512,self._cardinality,_strides=strides)

        # # conv4
        # for i in range(6):
        #     strides = (2, 2) if i == 0 else (1, 1)
        #     x = residual_block(x, 512, 1024,self._cardinality, _strides=strides)

        # # conv5
        # for i in range(3):
        #     strides = (2, 2) if i == 0 else (1, 1)
        #     x = residual_block(x, 1024, 2048,self._cardinality, _strides=strides)

        # x = layers.GlobalAveragePooling2D()(x)
        # x = layers.Dense(2)(x)

        # model = keras_Model(inputs=[y], outputs=[x])  

        # # set model. 
        # self._model = model
        # self._is_model_created = True

        # self._model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', 'categorical_accuracy'])
        # self._model.build(input_shape=(None, None, 3))
        # self._model.summary() ## TODO make this be called seperately.

        cnn_model = ResNet50(include_top=False, weights='imagenet', input_shape=None)
 
        
        final_network = Sequential()
        final_network.add(Lambda(lambda img: tf.image.resize_images(img,[self._img_width,self._img_height]), input_shape=(None,None,3)))
        final_network.add(Lambda(lambda img: tf.image.convert_image_dtype(img, tf.float32)))
        final_network.add(cnn_model)
        final_network.add(Flatten())
        final_network.add(Dense(2000, activation='relu'))
        final_network.add(Dropout(0.4))
        final_network.add(Dense(1000, activation='relu'))
        final_network.add(Dropout(0.4))
        final_network.add(Dense(500, activation='relu'))
        final_network.add(Dropout(0.4))
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

        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])
        self._model.build(input_shape=(None, None, 3))
        self._model.summary() ## TODO make this be called seperately.

        return final_network