from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
import pickle
from liveness.generic import AbstractModel
import os

class QualitySVMModel(AbstractModel):
    def __init__(self,logger):
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        svm = SVC(C=0.7, gamma='auto') # todo allow params to be set through constructor
        self._model = GridSearchCV(svm, parameters, cv=5)

        super().__init__(logger)

    def train(self, training_inputs, training_outputs):
        """Train the SVM model
        
        Arguments:
            training_inputs {np.array} -- Array of input vectors
            training_outputs {[type]} -- Array of expected outputs (fake/real encoded)
        """

        self._model.fit(training_inputs, training_outputs)

    def evaluate(self, input_img):
        return self._model.predict(input_img)

    def save(self, pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(self._model, f)
        
    def load(self, pickle_filename):
        with open(pickle_filename, 'rb') as f:
            self._model = pickle.load(f)

    def test(self, input_x, input_y):
        return self._model.score(input_x, input_y)

class QualityLDAModel(AbstractModel):
    def __init__(self):
        self._model = LDA()

        super().__init__(logger)

    def train(self, training_inputs, training_outputs):
        """Train the SVM model
        
        Arguments:
            training_inputs {np.array} -- Array of input vectors
            training_outputs {[type]} -- Array of expected outputs (fake/real encoded)
        """

        self._model.fit(training_inputs, training_outputs)

    def evaluate(self, input_img):
        return self._model.predict(input_img)

    def save(self, pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(self._model, f)
        
    def load(self, pickle_filename):
        with open(pickle_filename, 'rb') as f:
            self._model = pickle.load(f)

    def test(self, input_x, input_y):
        return self._model.score(input_x, input_y)