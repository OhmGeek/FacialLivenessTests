import logging

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from datasets.nuaa import NUAADataset
from datasets.replayattack import ReplayAttackDataset
from image_data_generator import ImageDataGenerator
from liveness.cnn.residual.model import ResidualNetwork
from preprocessing.face_extraction import pre_process_fn, preprocess_fn_all


def main():
    # First, fetch the two distinct sets of data.
    # Two neurons: [1.0, 0.0] -> fake, [0.0, 1.0] -> real
    # For each image in X, resize to (224,224) with 3 channels. Use OpenCV.

    # Now create the CNN model
    model = ResidualNetwork(logging.Logger("resnet"), learning_rate=0.0001, default_img_dimensions=(224, 224))

    # adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    # model.summary()

    # Now create the training set.
    dataset = NUAADataset(logging.getLogger("c.o.datasets.nuaa"), "/home/ohmgeek_default/datasets/nuaa")
    dataset.pre_process()

    imposter_set = dataset.read_dataset("ImposterRaw")
    imposter_y = np.tile([1.0, 0.0], (imposter_set.shape[0], 1))

    client_set = dataset.read_dataset("ClientRaw")
    client_y = np.tile([0.0, 1.0], (client_set.shape[0], 1))

    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    x, y = shuffle(x, y)

    # Train the model on our training set.
    batch_size = 256
    generator = ImageDataGenerator(x, y, batch_size=batch_size, preprocess_fn=pre_process_fn)

    # This is the validation generator
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"),
                                  "/home/ohmgeek_default/datasets/replayAttackDB/", mode='devel')
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")
    imposter_y = np.tile([1.0, 0.0], (imposter_set.shape[0], 1))

    client_set = dataset.read_dataset("real")
    client_y = np.tile([0.0, 1.0], (client_set.shape[0], 1))

    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    x, y = shuffle(x, y)
    print(x.shape)
    validation_generator = ImageDataGenerator(x, y, batch_size=batch_size, preprocess_fn=pre_process_fn)
    size_of_dataset = len(x)

    dataset = None
    x = None
    y = None
    imposter_set = None
    imposter_y = None
    client_set = None
    client_y = None

    # Now create the training set.
    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"),
                                  "/home/ohmgeek_default/datasets/replayAttackDB/", mode='test')
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")
    imposter_y = np.tile([1.0, 0.0], (imposter_set.shape[0], 1))

    client_set = dataset.read_dataset("real")
    client_y = np.tile([0.0, 1.0], (client_set.shape[0], 1))

    # Merge the two, and create the final sets.
    x = np.concatenate((imposter_set, client_set))
    y = np.concatenate((imposter_y, client_y))

    x, y = shuffle(x, y)

    test_generator = ImageDataGenerator(x, y, batch_size=8, preprocess_fn=pre_process_fn)

    model.fit_generator(generator, steps_per_epoch=size_of_dataset / batch_size, epochs=9, shuffle=True, verbose=1,
                        validation_data=validation_generator, validation_steps=len(x) / batch_size)
    model.save('alexnet.h5')

    score = model.test_generator(test_generator)
    print("Final Accuracy is: " + str(score))
    print("Shape going into eval", x.shape)
    print("Y shape going into eval", y.shape)
    print("x", x)

    imposter_set = dataset.read_dataset("attack")
    imposter_y = np.tile([0.0], imposter_set.shape[0])

    client_set = dataset.read_dataset("real")
    client_y = np.tile([1.0], client_set.shape[0])

    y = np.concatenate((imposter_y, client_y))

    y_pred = model.evaluate(preprocess_fn_all(x))

    y_pred = y_pred.argmax(axis=-1)

    print(y_pred)
    c_matrix = confusion_matrix(y, y_pred)

    print("Confusion matrix:")
    print(c_matrix)
    dataset.close()  # Important, to close the file.


if __name__ == "__main__":
    main()
