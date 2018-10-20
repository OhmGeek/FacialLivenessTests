from testframework.runner import TestRunner
import logging

def main():
    print("Running test.py")
    logging.basicConfig(level=logging.INFO)
    logger =
    logger.setLevel(20)
    test_runner = TestRunner(logging.getLogger("root.TestRunner"), [], [], [])
    test_runner.run()

if __name__ == "__main__":
    main()

