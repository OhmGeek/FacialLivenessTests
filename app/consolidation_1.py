import logging

import matplotlib.pyplot as plt
import numpy as np
from keras.backend import manual_variable_initialization

from datasets.replayattack import ReplayAttackDataset
from liveness.cnn.residual.model import ResidualNetwork
from liveness.quality.classifiers.lda import QualityLDAModel
from preprocessing.face_extraction import preprocess_fn_all, pre_process_fn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
def main():
    manual_variable_initialization(True)
    # first, set log level to display everything we want
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ryan/datasets/replay-attack/",
                                  mode='test')
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")
    client_set = dataset.read_dataset("real")
    output_imposter = [0.0 for x in range(imposter_set.shape[0])]
    output_client = [1.0 for x in range(client_set.shape[0])]

    input_x = np.concatenate((imposter_set, client_set))
    input_y = np.concatenate((output_imposter, output_client))

    input_x, input_y = shuffle(input_x, input_y)

    model_cnn = ResidualNetwork(logger)
    model_cnn.load('/home/ryan/Documents/dev/LivenessTests/models/cnn_v3.h5')

    model_wiqa = QualityLDAModel(logger)
    model_wiqa.load('/home/ryan/Documents/dev/LivenessTests/models/lda_model_v2.pkl')

    consolidation_input = []
    consolidation_output = []
    for i, img in enumerate(input_x):
        resnet_output = None
        wiqa_output = None
        try:

            resnet_output = model_cnn.evaluate(np.array([pre_process_fn(img)]))
            resnet_output = resnet_output[0, 1]
            wiqa_output = model_wiqa.evaluate(np.array([img]), get_probability=True)
            wiqa_output = wiqa_output[0, 1]
        except Exception:
            pass

        if resnet_output and wiqa_output:
            consolidation_input.append([resnet_output, wiqa_output])
            consolidation_output.append(input_y[i])


    # Now create a new classifier as the conslidation layer.

    consolidation_classifier = LDA(solver='eigen', shrinkage='auto')
    consolidation_classifier.fit(consolidation_input, consolidation_output)
    score = consolidation_classifier.score(consolidation_input[:200], consolidation_output[:200])
    print(score)

    test_input = consolidation_classifier.predict(consolidation_input[:200])
    mat = confusion_matrix(consolidation_output[:200], test_input)
    print("Confusion Matrix")
    print(mat)

    dataset.close()  # Important, to close the file.


if __name__ == "__main__":
    main()
