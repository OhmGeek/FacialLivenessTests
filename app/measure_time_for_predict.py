import logging
import time
import numpy as np
from keras.backend import manual_variable_initialization
from datasets.replayattack import ReplayAttackDataset
from liveness.cnn.residual.model import ResidualNetwork
from liveness.quality.classifiers.lda import QualityLDAModel
from preprocessing.face_extraction import pre_process_fn


def main():
    times_str = ""
    manual_variable_initialization(True)
    # first, set log level to display everything we want
    # TODO: change this to warn for production.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    dataset = ReplayAttackDataset(logging.getLogger("c.o.datasets.replayattack"), "/home/ryan/datasets/replay-attack/",
                                  mode='test')
    dataset.pre_process()

    imposter_set = dataset.read_dataset("attack")[:50]

    client_set = dataset.read_dataset("real")[:50]
    # Load the model.
    start_time = time.time()
    model = ResidualNetwork(logger)

    model.load('/home/ryan/Documents/dev/LivenessTests/models/cnn_v3.h5')
    stop_time = time.time()
    times_str += "time to load CNN model: " + str((stop_time - start_time)) + "\n"

    # Merge the data together.
    input_x = np.concatenate((imposter_set, client_set))

    resnet_times = []
    for img in input_x:
        start_time = time.time()
        model.evaluate(np.array([pre_process_fn(img)]))
        end_time = time.time()
        resnet_times.append(end_time - start_time)

    # Now we get the average time.
    average_resnet_time = sum(resnet_times) / len(resnet_times)
    times_str += "ResNet 1 image classification time: " + str(average_resnet_time) + "\n"

    ## Now do the same with the Quality Metric.
    model = None
    # Now we measure loading of LDA.
    start_time = time.time()
    model = QualityLDAModel(logger)
    model.load('/home/ryan/Documents/dev/LivenessTests/models/lda_model_v2.pkl')
    end_time = time.time()

    time_to_load_lda = end_time - start_time
    times_str += "LDA load time: " + str(time_to_load_lda) + "\n"

    lda_times = []
    for img in input_x:
        start_time = time.time()
        model.evaluate(np.array([img]))
        end_time = time.time()
        time_for_step = end_time - start_time
        lda_times.append(time_for_step)

    average_lda_time = sum(lda_times) / len(lda_times)

    times_str += "LDA predict time: " + str(average_lda_time) + "\n"

    dataset.close()  # Important, to close the file.

    print()
    print()
    print(times_str)


if __name__ == "__main__":
    main()
