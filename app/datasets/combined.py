from abc import ABC

from datasets.generic import Dataset


class NoDatasetsSpecifiedError(Exception):
    """ An exception to be thrown if no dataset is specified. """
    pass


class CombinedDataset(Dataset):
    """ Allows the processing of multiple datasets"""
    def __init__(self, datasets=None):
        """
        Create a combined dataset from a set of datasets.
        :param datasets: a list of datasets to be used for processing.
        """
        super().__init__()
        # Check whether datasets are actually defined.
        if datasets is None:
            raise NoDatasetsSpecifiedError()
        self._datasets = datasets

    def pre_process(self):
        raise NotImplementedError("Not yet implemented: pre_process on Combined dataset")

