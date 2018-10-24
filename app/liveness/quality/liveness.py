from liveness.generic import AbstractLivenessTest


class QualityLivenessTest(AbstractLivenessTest):
    def __init__(self, logger):
        super().__init__(logger)
