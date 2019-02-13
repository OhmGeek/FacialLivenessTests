import pickle

class ModelReader(object):
    def __init__(self):
        pass

    def read_from_file(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)