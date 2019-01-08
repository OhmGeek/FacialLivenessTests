from sklearn.svm import SVC

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
