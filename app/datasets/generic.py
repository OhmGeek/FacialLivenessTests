from abc import ABC, abstractmethod


class Dataset(ABC):
    """ A generic dataset """
    def __init__(self):
        pass

    @abstractmethod
    def pre_process(self):
        """
        Pre-process a dataset, to allow later processing.
        :return:
        """
        pass
