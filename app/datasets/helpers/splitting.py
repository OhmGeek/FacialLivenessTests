import cv2


class DataSplitter:
    def __init__(self, logger, vector_creator):
        self._logger = logger
        self._vector_creator = vector_creator

    def split(self, fake_set, real_set, percentage_train):
        """
        Take two tables of data, split them both by percentage, and return the result
        :param fake_set: The spoofed image set
        :param real_set: The real image set
        :param percentage_train: The percentage of data that should be training data
        :returns (train_vectors, train_outputs, test_vectors, test_outputs)
        """
        train_vectors = []
        train_outputs = []

        for imposter_img in fake_set:
            try:
                image = cv2.cvtColor(imposter_img, cv2.COLOR_BGR2GRAY)
                gaussian_image = cv2.GaussianBlur(image, (5, 5), 0)
                vector = self._vector_creator.create_vector(image, gaussian_image)
                train_vectors.append(vector)
                train_outputs.append(1.0)
            except:
                self._logger.error("Error while evaluating image.")

        for real_img in real_set:
            try:
                image = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
                gaussian_image = cv2.GaussianBlur(image, (5, 5), 0)
                vector = self._vector_creator.create_vector(image, gaussian_image)
                train_vectors.append(vector)
                train_outputs.append(0.0)
            except:
                self._logger.error("Error while evaluating image.")

        n = len(train_outputs)

        return (
            train_vectors[:int(percentage_train * n)],
            train_outputs[:int(percentage_train * n)],
            train_vectors[int(percentage_train * n) + 1:],
            train_outputs[int(percentage_train * n) + 1:]
        )
