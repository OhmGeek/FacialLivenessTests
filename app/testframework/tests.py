
class GenericTest(ABC):
    """ A generic implementation of a test """
    def __init__(self, logger):
        self._logger = logger