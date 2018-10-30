from datasets.nuaa import NUAADataset
from liveness.generic import DummyLivenessTest
from testframework.tests import TestDummyCase
from testframework.runner import TestRunner
import logging

def main():
    # first, set log level to display everything we want
    # TODO: change this to warn for production.
    logging.basicConfig(level=logging.INFO)

    print("Running test.py")
    # dataset = NUAADataset(logging.getLogger("c.o.datasets.nuaa"), "/home/ryan/datasets/nuaa")
    # dataset.pre_process()
    # data = dataset.get_all_datasets()

    models = [DummyLivenessTest(logging.getLogger("root.liveness.generic.DummyLivenessTest"))]
    tests = [TestDummyCase(logging.getLogger("c.o.testframework.tests.TestDummyCase"))]

    test_runner = TestRunner(logging.getLogger("c.o.testframework.runner"), models, tests, ["data"])

    test_runner.run()
    
if __name__ == "__main__":
    main()
