from abc import ABC, abstractmethod


class AbstractLivenessTest(ABC):
    def __init__(self, logger):
        self._logger = logger
        super().__init__()

    @abstractmethod
    def evaluate(self, input):
        """
            Given some input, run a test to assess the success of a model.

            :param input - a specific input image to test.
        """
        pass


class DummyLivenessTest(AbstractLivenessTest):
    def evaluate(self, input):
        # First, log the input to test.
        val = str(input)
        self._logger.info("Data input: %s" % val)

        # Return the input as the output for unit test sake.
        return input
