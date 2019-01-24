import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.engine.input_layer import Input

class ResidualNetwork(object):
    def __init__(self, logger, default_img_dimensions=(224,224), nb_channels=3, cardinality=32):
        self._logger = logger
        self._img_height = default_img_dimensions[1]
        self._img_width = default_img_dimensions[0]
        self._nb_channels = nb_channels
        self._cardinality = cardinality

        self._create_model()
        self._is_model_created = False

    def _create_model(self):
        # Now create input
        y = layers.Input(shape=(self._img_height, self._img_width, self._nb_channels))

        x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
        x = add_common_layers(x)

        # conv2
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        for i in range(3):
            project_shortcut = True if i == 0 else False
            x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

        # conv3
        for i in range(4):
            # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 256, 512, _strides=strides)

        # conv4
        for i in range(6):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 512, 1024, _strides=strides)

        # conv5
        for i in range(3):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 1024, 2048, _strides=strides)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1)(x)

        model = models.Model(inputs=[image_tensor], outputs=[x])  

        # set model. 
        self._model = model
        self._is_model_created = True
        
        return model