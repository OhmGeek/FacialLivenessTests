
class TestRunner:
    """ Run tests on a set of models """
    def __init__(self, logger, models, tests, data):
        self._logger = logger
        self._models = models
        self._tests = tests
        self._data = data

    def run():
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
                outcome = test.run(model, data)
                test_outcome.append(outcome)
            outcomes.append(test_outcome)
        self._logger.info("Finished logging")

        return outcomes