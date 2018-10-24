from datasets.nuaa import NUAADataset
from testframework.runner import TestRunner
import logging


def main():
    # print("Running test.py")
    # logging.basicConfig(level=logging.INFO)
    # test_runner = TestRunner(logging.getLogger("root.TestRunner"), [], [], [])
    # test_runner.run()
    dataset = NUAADataset("/home/ryan/datasets/nuaa")
    dataset.pre_process()


if __name__ == "__main__":
    main()
