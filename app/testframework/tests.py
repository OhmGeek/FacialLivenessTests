from abc import ABC


class GenericTest(ABC):
    """ A generic implementation of a test """

    def __init__(self, logger):
        """
        Create an instance of a generic test.
        :param logger: A logging object for a given test.
        """
        self._logger = logger
