from abc import ABC

import torch


class KineticEnergy(ABC):

    '''
    An abstract base class for implementing the key functionality of KineticEnergy for hamiltonial monte carlo

    Methods:
    ----------------------------------------------------------------
    get_ke (q,p) : return the kinetic energy given position and momentum
    dkdp (p, K): return the gradient of K w.r.t to p
    dkdq (q, K): return the gradient of K w.r.t to q
    '''
    def __init__(self):
        pass
    
    def log_prob(self, q, p): 
        pass

    def dkdp(self, p): 
        pass

    def dkdq(self, q): 
        pass
