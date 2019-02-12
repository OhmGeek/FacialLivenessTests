import pickle

class ModelWriter(object):
    def __init__(self, model):
        self._model = model
    
    def write_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._model, f)

