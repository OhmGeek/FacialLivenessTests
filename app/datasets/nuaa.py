from datasets.generic import Dataset
import numpy as np
import cv2
import glob


class NUAADataset(Dataset):
    _filename = ...  # type: str

    def __init__(self, filename):
        """
        Initialise the dataset
        :param filename: the location of the extracted dataset on disk (the root folder).
        """
        self._filename = filename
        super().__init__()

    def pre_process(self):
        """Pre-process the dataset."""
        # First, load imposter images.

        imposter_images = []

        for img_filename in glob.iglob(self._filename + 'ImposterRaw/**/*.jpg'):
            img = cv2.imread(img_filename, mode='RGB')
            imposter_images.append(img)

