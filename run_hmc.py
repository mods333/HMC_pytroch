import torch
from KineticEnergy.EucledianGaussian import EucledianGaussian
from PotentialEnergy.NormalPotential import NormalPotential
from Integrator.leapFrog import leapFrog
from HMC.hmc import hmcSampler

import matplotlib.pyplot as plt

def main():
    
    dim = 2
    num_samples = 1000
    ke = EucledianGaussian(torch.eye(dim))
    pe = NormalPotential(torch.ones(dim), torch.eye(dim))

    sampler = hmcSampler(num_samples, ke, pe, leapFrog,10, 0.01, 20)

    q_init = torch.zeros(dim)
    samples = sampler.sample(q_init)

    samples = torch.stack(samples)

    fig = plt.figure()
    true_samples = torch.stack([pe.sample() for _ in range(num_samples)])
    
    plt.subplot(2, 1, 1)
    plt.scatter(samples[:,0].numpy(), samples[:,1].numpy())
    
    plt.subplot(2, 1, 2)
    plt.scatter(true_samples[:,0].numpy(), true_samples[:,1].numpy())
    
    plt.savefig('gaussian_samples.png')

if __name__ == '__main__':
    main()