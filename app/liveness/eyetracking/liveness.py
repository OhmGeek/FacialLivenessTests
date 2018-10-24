from liveness.generic import AbstractLivenessTest


class EyetrackingLivenessTest(AbstractLivenessTest):
    def __init__(self, logger):
        super().__init__(logger)
