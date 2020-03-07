import torch
from torch.distributions import multivariate_normal


class EucledianGaussian():

    def __init__(self,sigma):

        self.ke = multivariate_normal.MultivariateNormal(torch.zeros(sigma.shape[0]), sigma)
        self.inv_sigma = torch.inverse(sigma)

    def sample(self):
        return self.ke.rsample()

    def log_prob(self, q, p):
        return self.ke.log_prob(p)

    def dkdp(self, p):
        return torch.matmul(self.inv_sigma, p)
    
    def dkdq(self, q):

        #return dk/dq 
        return torch.zeros(self.inv_sigma.shape[0])
