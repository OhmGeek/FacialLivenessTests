from datasets.nuaa import NUAADataset
from testframework.runner import TestRunner
import logging

def main():
    # print("Running test.py")
    # logging.basicConfig(level=logging.INFO)
    # test_runner = TestRunner(logging.getLogger("root.TestRunner"), [], [], [])
    # test_runner.run()
    dataset = NUAADataset(logging.getLogger("c.o.datasets.nuaa"), "/home/ryan/datasets/nuaa")
    dataset.get_all_datasets()
    dataset.pre_process()
    dataset.get_all_datasets()
    
if __name__ == "__main__":
    main()
