import torch
from torch.distributions import multivariate_normal


class NormalPotential():
    '''
    Class implementing a gaussian porential energy
    '''

    def __init__(self,mean, std):

        self.distribution = multivariate_normal.MultivariateNormal(mean, std)

    def sample(self):
        #generally this will not be implemented as this is the purpose of HMC, but for a guassian we have other methods to sample. This function is just for sanity check
        return self.distribution.sample()

    def log_prob(self,value):

        self.log_prob_value = self.distribution.log_prob(value)
        return self.log_prob_value

    def dvdq(self, value):

        value = value.clone().detach().requires_grad_(True)
        log_prob_value = self.log_prob(value).clone().requires_grad_(True)
        return torch.autograd.grad(log_prob_value, value)[0]

if __name__ == "__main__":
    temp = NormalPotential(torch.ones(2), torch.eye(2))

    print("The log prob at {} is : {}".format(torch.zeros(2), temp.log_prob(torch.zeros(2))))
    print("The gradient is {}".format(temp.dvdq(torch.ones(2))))
