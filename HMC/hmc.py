import torch

from tqdm import tqdm


class hmcSampler():

    def __init__(self, num_samples, kineticEnergy, potentialEnergy, integrator, pathlength, stepsize, maxsteps):
        '''
        A wrapper for sampling using HMC
        Arguments:
            num_samples: Number of samples to be generated
            KinteicEnergy: An object specifying the kinetic energy and its gradients
            PotentialEnergy: An object specifying the potential energy and its gradients
            pathlength: Specify the length of the trajectory over which to integrate
            stepsize: The required step size
            maxsteps: The maximum number of steps to be taken during the integration
        '''

        self.num_samples = num_samples
        self.kineticEnergy = kineticEnergy
        self.potentialEnergy = potentialEnergy
        self.integrator = integrator
        self.pathlength = pathlength
        self.stepsize = stepsize
        self.maxsteps = maxsteps

    def sample(self, q_init):
        '''
        A method that generated self.num_samples
        Arguments: 
            q_init: An initialization of the position
        '''

        num_reject = 0
        samples = [q_init]

        for _ in tqdm(range(self.num_samples)):
            
            p_init = self.kineticEnergy.sample().clone()

            q_new, p_new = self.integrator(samples[-1].clone(),
                                            p_init, 
                                            self.kineticEnergy.dkdp,
                                            self.kineticEnergy.dkdq,
                                            self.potentialEnergy.dvdq,
                                            self.pathlength,
                                            self.stepsize,
                                            self.maxsteps
                                            )    
            
            rho = min(1, self.compute_acceptance_ratio(samples[-1], p_init, q_new, p_new))
            
            if torch.log(torch.rand(1)) <= rho:
                samples.append(q_new)
            else:
                num_reject += 1
                samples.append(samples[-1].clone())                

        print("Acceptance ratio is {:.2f}".format(1 - num_reject/self.num_samples))
        
        return samples

    def compute_acceptance_ratio(self, q0, p0, q_new, p_new):

        Hq0p0 = -self.potentialEnergy.log_prob(q0) - self.kineticEnergy.log_prob(q0, p0)  
        Hq_new_p_new = -self.potentialEnergy.log_prob(q_new) - self.kineticEnergy.log_prob(q_new, p_new)

        return Hq0p0 - Hq_new_p_new 
