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

    print("Trained.")

    model.save('pretrained-voxnet.pkl')

    print("Saved.")

    # clear and empty the memory.
    voxels = None
    labels = None

    print("Now ready to load testing set")
    # Now test using the test dataset.
    voxels, labels = helper.load_data_from_npy(npy_dir, mode='testing')

    print(voxels)

    print("Now labels:")
    print(labels)    

    labels = to_categorical(labels)
    results = model.test(voxels, labels)

    print(results)


if __name__ == "__main__":
    train()