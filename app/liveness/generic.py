from abc import ABC, abstractmethod


class AbstractLivenessTest(ABC):
    def __init__(self, logger):
        self._logger = logger
        super().__init__()
