"""
This file uses the VRN neural network to convert a 2D image into a 3D voxel structure.

This is then passed for classification.
"""
import numpy as np
import cv2
from keras.models import load_model

from liveness.vox.reconstruction import custom_layers

def preprocess_image_for_builder(image, new_size=(192,192)):
    # This takes an image, preprocesses the image.
    ## First, resize to new size (default: 192 x 192)
    image = cv2.resize(image, new_size)

    ## B,G,R -> R,G,B
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    image = np.swapaxes(image, 2, 0)
    image = np.swapaxes(image, 2, 1)

    # R,G,B -> image
    image = np.array([image])

    return image

class FaceVoxelBuilder(object):
    def __init__(self, logger, reconstructor_path=None):
        self._logger = logger
        
        # Load in the keras model
        if(reconstructor_path is None):
            reconstructor_path = 'vrn-unguided-keras.h5'
        self._reconstructor_path = reconstructor_path

        # build the model to use later
        self._model = self._create_builder_model(reconstructor_path)

    def _create_builder_model(self, reconstructor_path):
        """
            Create the 3D reconstruction keras model.

            - reconstructor_path: the path to the model on disk
        """

        custom_objects = {
            'Conv': custom_layers.Conv,
            'BatchNorm': custom_layers.BatchNorm,
            'UpSamplingBilinear': custom_layers.UpSamplingBilinear
        }

        model = load_model(reconstructor_path, custom_objects=custom_objects)

        return model
    
    def build_3d(self, image):
        """
            Reconstruct a face from 2D -> 3D
        """
        img = preprocess_image_for_builder(image)

        pred = self._model.predict(img)
        vol = pred[0] * 255
        im = img[0]
        im = np.swapaxes(im, 0, 1)
        im = np.swapaxes(im, 1, 2)

        vol_rgb = np.stack(((vol > 1) * im[:,:,0],
                        (vol > 1) * im[:,:,1],
                        (vol > 1) * im[:,:,2]), axis=3)

        return vol_rgb