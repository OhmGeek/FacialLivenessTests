import logging
import keras
import numpy as np

from liveness.vox.reconstruction.vrn import FaceVoxelBuilder
from preprocessing.face_extraction import preprocess_fn_all


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
        x_values = self.x[index * self.batch_size: (index + 1) * self.batch_size]
        y = self.y[index * self.batch_size: (index + 1) * self.batch_size]

        # First, extract faces.
        x = preprocess_fn_all(x_values)

        # Then find voxel representation.
        x = self.voxel_builder.build_3d_multiple(x)
        return x, y


class ImageDataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=32, shuffle=False, preprocess_fn=lambda x: x):
        self.batch_size = batch_size
        self.y = y
        self.x = x
        self.preprocess_fn = preprocess_fn
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        x_values = self.x[index * self.batch_size: (index + 1) * self.batch_size]
        y = self.y[index * self.batch_size: (index + 1) * self.batch_size]

        # First, extract faces.
        outputs = []
        for image in x_values:
            output = self.preprocess_fn(image)
            outputs.append(output)

        outputs = np.array(outputs)

        # Then find voxel representation.
        return outputs, y
