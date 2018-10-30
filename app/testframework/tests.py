from abc import ABC, abstractmethod
from liveness.generic import AbstractLivenessTest

class ModelNotValidError(Exception):
    pass


class TestDummyCase(GenericTest):
        

class GenericTest(ABC):
    """ A generic implementation of a test """

    def __init__(self, logger):
        """
        Create an instance of a generic test.
        :param logger: A logging object for a given test.
        """
        self._logger = logger

    def run(self, model, data):
        # First, make sure the given model is actually a liveness test. If not, break.
        if not isinstance(model, typeof(AbstractLivenessTest)):
            raise ModelNotValidError()

        # get the model output
        output = model.evaluate(data)

        # return outcome from the model.
        return output