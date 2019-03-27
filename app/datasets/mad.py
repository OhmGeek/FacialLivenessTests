from datasets.generic import Dataset
import numpy as np
import cv2
import glob
import h5py
from os.path import isfile as check_file_exists
from os.path import dirname, realpath
import random

class PreprocessingIncompleteError(Exception):
    """ An error to be raised when preprocessing hasn't been executed """
    pass

# TODO refactor some of the methods into the base class.
class MaskAttackDataset(Dataset):
    # TODO export labels to some map (so we can understand what real/attack are in terms of other datasets.)
    def __init__(self, logger, filename):
        """
        Initialise the dataset
        :param filename: the location of the extracted dataset on disk (the root folder).
        :param mode: the mode (e.g. devel, test, etc)
        :param labels: the output labels we expect (real, attack)
        """
        self._logger = logger
        self._filename = filename
        self._is_file_open = False
        self._datasets = None
        self._output_filename = dirname(realpath(__file__)) + '/../../datasets/mad/mad.h5'
        self._labels = ["B", "C"]
        super().__init__()

    def pre_process(self):
        """Pre-process the dataset."""
        # First, load imposter images.
        self._datasets = []

        # If a training set exists, don't try to start again from scratch.
        if check_file_exists(self._output_filename):
            self._logger.info("File exists, so finish preprocessing early.")
            return

        # TODO: go through preprocessing dataset into h5 file (all in one).

        with h5py.File(self._output_filename, 'w') as hf:
            for label in self._labels:
                self._logger.info("Start looking at label %s in dataset." % label)
                label_images = []
                # Go through each h5 file
                glob_arg = self._filename + '/%s/**/*.hdf5' % label
                print(glob_arg)
                for vid_filename in glob.iglob(glob_arg, recursive=True):
                    print(vid_filename)
                    # Now open h5py file for reading.
                    with h5py.File(vid_filename, "r") as mad_file:
                        file_images = mad_file['Color_Data']
                        for img in file_images[0:5]:
                            # First, we need to transpose and reorder for Opencv.
                            img = img.transpose(2,1,0)
                            label_images.append(img)

                # save to the dataset.
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
    

