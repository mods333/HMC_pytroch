
def leapFrog(q, p, dkdp, dkdq, dvdq, pathlength, stepsize, maxsteps):
    '''
    LeapFrog integrator for hamiltoninan montecarlo

    Paramters:
        q: (torch.tensor) Inital position
        p: (torch.tensor) Inital momentum
        dkdp : Function that return the gradient of the kinetic energy w.r.t to momentum
        dkdq : Function that return the gradient of the kinetic energy w.r.t to position
        dvdq : Function that return the gradient of the potentioal energy w.r.t to position
        pathlength: (torch.FloatTensor) The length of the trjactory to integrate over
        stepsize: (float) The step size for integration
        maxsteps: (int) Maximum number of steps for which q and p should be updated
    Return: 
        q, p : New position and momentum
    '''

    p -= (stepsize/2)*(dkdq(p) + dvdq(p))
    
    numsteps = min(maxsteps, pathlength//stepsize) 
    for _ in range(numsteps-1):

        q += stepsize*dkdp(p)
        p -= stepsize*(dkdq(p) + dvdq(p))

    q += stepsize*dkdp(p)
    p -= (stepsize/2)*(dkdq(p) + dvdq(p))

    return q, -p #Flipping the sign of the momentum makes the Hamiltonian Transition reversible