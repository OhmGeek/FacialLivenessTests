import numpy as np
import keras
from liveness.vox.reconstruction.vrn import FaceVoxelBuilder
from PIL import Image
import face_recognition
import logging

class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=32, shuffle=False):
        self.batch_size = batch_size
        self.y = y
        self.x = x
        self.voxel_builder = FaceVoxelBuilder(logging.Logger("VoxelBuilder"))
        self.on_epoch_end()


    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        x_values = self.x[index*self.batch_size : (index+1)*self.batch_size]
        y = self.y[index*self.batch_size : (index + 1)*self.batch_size]

        # TODO convert x_values to voxels.
        x = self.voxel_builder.build_3d_multiple(x_values)

        return x, y

