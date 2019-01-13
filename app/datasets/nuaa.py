from datasets.generic import Dataset
import numpy as np
import cv2
import glob
import h5py
from os.path import isfile as check_file_exists
from os.path import dirname, realpath

class PreprocessingIncompleteError(Exception):
    """ An error to be raised when preprocessing hasn't been executed """
    pass


class NUAADataset(Dataset):

    def __init__(self, logger, filename, labels=["ImposterRaw", "ClientRaw"]):
        """
        Initialise the dataset
        :param filename: the location of the extracted dataset on disk (the root folder).
        """
        self._logger = logger
        self._filename = filename
        self._labels = labels
        self._is_file_open = False
        self._datasets = None
        self._output_filename = dirname(realpath(__file__)) + '/../../datasets/nuaa/nuaa.h5'
        super().__init__()

    def pre_process(self):
        """Pre-process the dataset."""
        # First, load imposter images.
        self._datasets = []

        # If a training set exists, don't try to start again from scratch.
        if check_file_exists(self._output_filename):
            self._logger.info("File exists, so finish preprocessing early.")
            return

        with h5py.File(self._output_filename, 'w') as hf:
            for label in self._labels:
                self._logger.info("Start looking at label %s in dataset." % label)
                label_images = []

                # For each label, go through all the files in the directory/classification.
                for img_filename in glob.iglob(self._filename + '/%s/**/*.jpg' % label):
                    img = cv2.imread(img_filename)
                    label_images.append(img)
                # Now images are created and stored in the dataset.
                dataset = hf.create_dataset(label, data=label_images)
                self._logger.info("Dataset created")
                self._datasets.append(dataset)

        self._logger.info("Dataset preprocessing completed.")

    def get_all_datasets(self):
        if(self._datasets == None):
            self._logger.error("Preprocessing wasn't executed. Therefore, no datasets are loaded.")
            raise PreprocessingIncompleteError()
        return self._datasets

    def _open_file(self):
        self._h5_file = h5py.File(self._output_filename, 'r')
        self._is_file_open = True

    def read_dataset(self, label):
        if(self._datasets is None):
            self._logger.error("Preprocessing not executed.")
            raise PreprocessingIncompleteError()

        output = None
        
        if(not self._is_file_open):
            self._open_file()
        
        output = self._h5_file[label]
        
        return output

    def close(self):
        if(self._is_file_open):
            self._h5_file.close()
    

