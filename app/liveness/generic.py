from abc import ABC, abstractmethod
import pickle

class AbstractModel(ABC):
    def __init__(self, logger):
        self._logger = logger

        # Generic properties that we need.
        #
        #  - self._model : this is the model object that's pickled. One MUST use this.
        
        super().__init__()

    # Methods that need to be implemented.
    @abstractmethod
    def train(self, x, y):
        """Train the model using the training data
        
        Arguments:
            x {np.ndarray} -- Contains the input data (see images)
            y {[type]} -- Contains the classification results (see fake/live)
        """
        pass

    @abstractmethod
    def test(self, x, y):
        """Test the model using test data
        
        Arguments:
            x {np.ndarray} -- Array of images to test on.
            y {np.ndarray} -- Array of expected outputs for the classifier
        """

        pass

    @abstractmethod
    def evaluate(self, input_img):
        """Get the model output of a specified image
        
        Arguments:
            input_img {np.ndarray} -- Input image, to fetch the classification for.
        """
        pass
    
    # -- Standard methods that are the same throughout
    def save(self, pickle_path):
        """Save the model to file
        
        Arguments:
            pickle_path {string} -- The path to save the file to
        """

        with open(pickle_path, 'wb') as f:
            pickle.dump(self._model, f)
            
    def load(self, pickle_filename):
        """Load a model from file
        
        Arguments:
            pickle_filename {string} -- The path to load the model from
        """

        with open(pickle_filename, 'rb') as f:
            self._model = pickle.load(f)


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
