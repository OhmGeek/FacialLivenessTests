import keras
import helper
from liveness.vox.classification.voxnet import VoxNet
import logging
from keras.utils import to_categorical
def train():
    npy_dir = '/home/ryan/datasets/suod'

    model = VoxNet(logging.getLogger("VoxNet"))

    # Training, using the prebuild loader with our model.
    voxels, labels = helper.load_data_from_npy(npy_dir, mode='training')
    labels = to_categorical(labels)
    print(labels)
    model.train(voxels, labels)

    model.save('pretrained-voxnet.pkl')

    # Now test using the test dataset.
    voxels, labels = helper.load_data_from_npy(npy_dir, mode='testing')
    labels = to_categorical(labels)
    results = model.test(voxels, labels)

    print(results)


if __name__ == "__main__":
    train()