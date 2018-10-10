# Using keras, this script trains and tests our model as an experiment.
# It generates an autoencoder which we can then use to find an actual face.

# In our later system, we'll then take the vector output, run an SQL query against the vector
# (or an equivalent for a database of our choice).
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

face_dataset = sklearn.datasets.fetch_olivetti_faces()

def createModel():
    # this is the size of our encoded representations
    encoding_dim = 15

    # this is our input placeholder
    input_img = Input(shape=(4096,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
     
    return model


