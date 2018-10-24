from datasets.generic import Dataset
import numpy as np
import cv2
import glob
import h5py


class NUAADataset(Dataset):
    _filename = ...  # type: str

    def __init__(self, filename, labels=["ImposterRaw"]):
        """
        Initialise the dataset
        :param filename: the location of the extracted dataset on disk (the root folder).
        """
        self._filename = filename
        self._labels = labels
        super().__init__()

    def pre_process(self):
        """Pre-process the dataset."""
        # First, load imposter images.
        # Shape of image (N, max(H), max(W), channels).
        # N: number of images, channels: number of channels in image e.g. 3 for RGB,
        # H: height of all images. We use the maximium and we should probably mask.
        # W: width for all images. We use max, but should probably mask.
        print("start preprocess")
        imposter_images = []
        max_ctr = 0
        for label in self._labels:
            
            for img_filename in glob.iglob(self._filename + '/%s/**/*.jpg' % label):
                print(img_filename)
                img = cv2.imread(img_filename)
                imposter_images.append(img)
                max_ctr += 1

                # Break for testing TODO: Remove.
                if max_ctr > 1000:
                    break
            with h5py.File('nuaa.h5', 'w') as hf:
                dset = hf.create_dataset(label, data=imposter_images)
                print("Shape Size:", dset.shape)
