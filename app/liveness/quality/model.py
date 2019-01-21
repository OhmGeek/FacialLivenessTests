from sklearn.svm import SVC
import pickle
import os

class QualitySVMModel(object):
    def __init__(self):
        self._model = SVC() # todo allow params to be set through constructor

    def train(self, training_inputs, training_outputs):
        """Train the SVM model
        
        Arguments:
            training_inputs {np.array} -- Array of input vectors
            training_outputs {[type]} -- Array of expected outputs (fake/real encoded)
        """

        self._model.fit(training_inputs, training_outputs)

    def evaluate(self, input_img):
        return self._model.predict(input_img)

    def save(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, 'model.pkl'), 'wb') as f:
            pickle.dump(self._model, f)
        

    def test(self, input_x, input_y):
        return self._model.score(input_x, input_y)

