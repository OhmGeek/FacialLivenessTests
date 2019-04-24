import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from liveness.generic import AbstractModel
from liveness.quality.metric_generator import preprocessor


class QualityLDAModel(AbstractModel):
    def __init__(self, logger):
        self._model = LDA(shrinkage='auto', solver='eigen')

        super().__init__(logger)

    def train(self, training_inputs, training_outputs):
        """Train the SVM model

        Arguments:
            training_inputs {np.array} -- Array of input vectors
            training_outputs {[type]} -- Array of expected outputs (fake/real encoded)
        """
        training_inputs, training_outputs = preprocessor(training_inputs, training_outputs, self._logger)

        self._model.fit_transform(training_inputs, training_outputs)

    def evaluate(self, input_img, get_probability=False):
        training_inputs, _ = preprocessor(input_img, None, self._logger)

        if get_probability:
            return self._model.predict_proba(training_inputs)

        return self._model.predict(training_inputs)

    def save(self, pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(self._model, f)

    def load(self, pickle_filename):
        with open(pickle_filename, 'rb') as f:
            self._model = pickle.load(f)

    def test(self, input_x, input_y):
        print("Shape of test set:")
        print("    input_x", input_x.shape)
        print("    input_y", input_y.shape)
        training_inputs, training_outputs = preprocessor(input_x, input_y, self._logger)
        return self._model.score(training_inputs, input_y)