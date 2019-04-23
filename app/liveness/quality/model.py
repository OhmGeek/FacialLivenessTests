from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
import pickle
from liveness.generic import AbstractModel
import os
from liveness.quality.metrics.factory import metric_factory
from liveness.quality.metric_vector import DefaultMetricVectorCreator
import cv2
import numpy as np

def preprocessor(data, outputs, logger):
    metrics_names = [
        "ad",
        "biqi",
        "gme",
        "gpe",
        "hlfi",
        "jqi",
        "lmse",
        "mams",
        "mas",
        "md",
        "mse",
        "nae",
        "niqe",
        "nxc",
        "psnr",
        "ramd",
        "sc",
        "sme",
        "snr",
        "spe",
        "ssim",
        "tcd",
        "ted",
        "vifd"
    ]

    metrics = metric_factory(metrics_names, logger)
    vector_creator = DefaultMetricVectorCreator(metrics)
    train_vectors = []
    train_outputs = []
    for i in range(len(data)):
        try:
            client_img = data[i]
            image = cv2.cvtColor(client_img, cv2.COLOR_BGR2GRAY)

            gaussian_image = cv2.GaussianBlur(image,(5,5),0)
            vector = vector_creator.create_vector(image, gaussian_image)
            # None can't be in the vector. Everything must be a number.
            if None in vector:
                raise Exception()

            train_vectors.append(vector)

            if outputs is not None:
                train_outputs.append(outputs[i])
        except Exception as e:
            logger.error("Error while evaluating image")
            print(e)
            raise e
    return train_vectors, train_outputs

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
