# Author: Ethan Nanavati
# Last Edited: 6/28/24
# Known Bugs: None

from Dataset_Management_Class import GMMDataset
import numpy as np

class Metropolis(object):
    ''' Sampler for performing Metropolis MCMC with Random Walk proposals

    Requires a target random variable with continuous real sample space.
    
    Example Usage
    -------------
    >>> import matplotlib.pyplot as plt

    # Create the target distribution
    >>> dataset = GMMDataset(2,[-10,10],"GG")
    >>> dataset.Plot_Distribution()

    # Create the sampler
    >>> MHO = Metropolis(dataset,[-10,10],42)

    # Get samples from sampler
    >>> samples = MHO.GetSamples(p_width=8)

    # visualize the samples from Metropolis sampling
    >>> plt.hist(samples,bins=20)
    >>> plt.show()


    Attributes
    ----------
    distribution : GMMDataset object
        Given a value of random variable, computes the pdf using the "Evaluate_Scaled" method of distribution    
    lower: int
        specifies the lower bound for the means used in the GMM for distribution
    upper: int
        specifies the upper bound for the means used in the GMM for distribution
    random_state : numpy.random.RandomState
        Pseudorandom number generator, supports .rand() and .randn()
    '''
    # consider using a seperate argument that les me change the proposal function
    # make sure that num_samples is bigger than 2000 (for burnin)
    def __init__(self, distribution, bounds, Random_State=42):
        ''' Constructor for RandomWalkSampler object

        Args
        ----
        distribution : GMMDataset object
            Given a value of random variable, computes the pdf
        bounds : 2 entry array of size (2,)
            First entry represents the lower bound for the means, second represents the upper bound
            The data will fall outside these bounds, but this gives some data for initialization
        random_state : int
            Initial state of this sampler's random number generator.
            Set deterministically for reproducability and debugging.
            will create a numpy PRNG with that int as seed.

        Returns
        -------
        New Metropolis object
        '''
        self.distribution = distribution
        self.Random_State = np.random.RandomState(int(Random_State))
        self.lower = bounds[0]
        self.upper = bounds[1]



    def GetSamples(self, burnin = 5000, keep = 5000, p_width = 1):
        ''' Draw samples from target distribution via MCMC

        Args
        ----
        keep : int
            Number of samples to generate that we intended to keep
        burnin : int
            Number of samples to generate that we intend to discard
        p_width : int
            Standard deviation for the proposal width. It controls how far away from z_curent we look for z_prime

        Returns
        -------
        keep_chain : list of ints
            Each entry is a sample from the MCMC procedure.
            Will contain keep entries.
        '''

        a = self.lower
        b=self.upper

        # choose a random number uniformly from [a,b]
        initial = self.Random_State.rand() * (b - a) - a
        z_curent = initial

        # burn samples
        keep_chain = []
        for i in range(burnin + keep):
            z_prime = self.DrawProposal(z_curent, p_width)
            accept_ratio = self.distribution.Evaluate_Scaled(z_prime) / self.distribution.Evaluate_Scaled(z_curent)
            accept_threshold = self.Random_State.rand()

            if accept_ratio >= accept_threshold:
                z_curent = z_prime
            # implicitly, if accept_ratio < accept_threshold, z_curent remains unchanged
            if i >= burnin:
                keep_chain.append(z_curent)
        self.keep_chain = keep_chain # store for future error analysis
        return keep_chain            


    def DrawProposal(self, center, p_width):
        ''' Draw sample z_prime from the proposal distribution centered at center

        Args
        ----
        center : int
            mean of the proposal distribution.
            This will be the current value in the Markov chain (z_curent)
        p_width : int
            Standard deviation for the proposal width. It controls how far away from z_curent we look for z_prime

        Returns
        -------
        z_prime : int
            a sample from the proposal distribution centered at the current value in the Markov chain.
            proposal distribution is a normal dist
        '''
        mean = center
        stdev = p_width
        z_prime = self.Random_State.normal(mean,stdev)
        return z_prime



    


# import matplotlib.pyplot as plt

# dataset = GMMDataset(2,[-10,10],"GG")
# dataset.Plot_Distribution()
# MHO = Metropolis(dataset,[-10,10],42)
# samples = MHO.GetSamples(p_width=8)
# plt.hist(samples,bins=20)
# plt.show()