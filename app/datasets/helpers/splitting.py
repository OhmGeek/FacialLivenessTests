class DataSplitter:
    def __init__(self):
        pass

    def split(self, dataset, percentage_divisions=None):
        """
        Divide up a specified dataset into the specified percentages.
        :param dataset:
        :param percentage_divisions: list of the percentages to divide
        :return: A len(percentage_divisions) tuple containing the divided data.
        """
        # First, use the default of returning the entire dataset (no split).
        if percentage_divisions is None:
            percentage_divisions = [100]
        pass


