class TestRunner:
    """ Run tests on a set of models """

    def __init__(self, logger, models, tests, data):
        """
        Create an instance of a test running
        :param logger: A logging object to log the execution.
        :param models: A list of models to test.
        :param tests: A list of tests to execution on the models.
        :param data: A set of data to use.
        """
        self._logger = logger
        self._models = models
        self._tests = tests
        self._data = data

    def run(self):
        """ 
        run a set of tests, on a set of models, with a set of datasets 
        
        Returns a list of outcomes, per test, per model.
        """
        self._logger.info("Starting the test execution")
        outcomes = []
        test_counter = 0
        for test in self._tests:
            test_counter += 1
            self._logger.info("Executing test number %d of %d" % (test_counter, len(self._tests)))
            test_outcome = []
            model_counter = 0
            for model in self._models:
                self._logger.info("Executing for model %d of %d" % (model_counter, len(self._models)))
                outcome = test.run(model, self._data)
                test_outcome.append(outcome)
            outcomes.append(test_outcome)
        self._logger.info("Finished logging")

        return outcomes
