import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from liveness.generic import AbstractModel
from liveness.quality.metric_generator import preprocessor


class QualitySVMModel(AbstractModel):
    def __init__(self, logger):
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svm = SVC(C=0.7, gamma='auto')  # todo allow params to be set through constructor
        self._model = GridSearchCV(svm, parameters, cv=5)

        super().__init__(logger)

    def train(self, training_inputs, training_outputs):
        """Train the SVM model

        Arguments:
            training_inputs {np.array} -- Array of input vectors
            training_outputs {[type]} -- Array of expected outputs (fake/real encoded)
        """
        training_inputs, training_outputs = preprocessor(training_inputs, training_outputs, self._logger)
        self._model.fit(training_inputs, training_outputs)

    def evaluate(self, input_img):
        training_inputs = preprocessor(input_img, self._logger)
        return self._model.predict(training_inputs)

    def save(self, pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(self._model, f)

    def load(self, pickle_filename):
        with open(pickle_filename, 'rb') as f:
            self._model = pickle.load(f)

    def test(self, input_x, input_y):
        training_inputs = preprocessor(input_x, self._logger)
        return self._model.score(training_inputs, input_y)