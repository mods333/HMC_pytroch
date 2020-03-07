from abc import ABC

import torch


class PotentialEnergy(ABC):
    '''An abstract base class for Potential Energy'''

    def __init_(self, *args, **kwargs):
        pass

    def log_prob(self, *args, **kwargs):
        pass

    def dvdq(self, q, V):
        pass
