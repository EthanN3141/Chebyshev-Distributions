# Author: Ethan Nanavati
# Last Edited: 6/28/24
# Known Bugs: None

from Metropolis_Class import Metropolis
from Chebyshev_Class import ChebyshevInterpolant
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import sympy




class Comparison:
    ''' Provides tools to compare Metropolis histogram and Chebyshev interpolation methods
    
    Example Usage
    -------------
    >>> import numpy as np

    >>> from Dataset_Management_Class import GMMDataset
    >>> from MetropolisHastings_Class import Metropolis
    >>> from Chebyshev_Class import ChebyshevInterpolant
    >>> from method_comparison import Comparison


    >>> # set hyperparameters
    >>> means_bounds = [-10,10] # the gaussians in the distribution will have means in this range
    >>> integration_bounds = [-20,20] # captures at least 99.96% of the mass with means=[-10,10]
    >>> num_peaks = 8 # gausian mixture will have this many means
    >>> burn_samples = 5000
    >>> keep_samples = 5000
    >>> num_bins = 3 * int(1 + np.ceil(3.322 * np.log(keep_samples))) # number of bins chosen via Sturge's rule: K = 1 + 3. 322 logN.
    >>> num_interpolating_points = 28
    >>> proposal_widths = [5] #[1,5,10]
    >>> random_state = 55 # which random state to use


    >>> # Create the distributions
    >>> distribution_GC = GMMDataset(num_peaks,means_bounds, 'GC',Random_State=random_state)
    >>> distribution_GE = GMMDataset(num_peaks,means_bounds, 'GE',Random_State=random_state)
    >>> distribution_GG = GMMDataset(num_peaks,means_bounds, 'GG',Random_State=random_state)
    >>> distributions = [(distribution_GC, "GC"), (distribution_GE, "GE"), (distribution_GG, "GG")] # to be looped through


    >>> # performs analysis on the 2 methods
    >>> storage_for_plotting = [] # stores info for the graphs at the end
    >>> for distribution in distributions:
    >>>     for p_width in proposal_widths:  
    >>>         # Initialize the Metropolis sampler  
    >>>         sampler = Metropolis(distribution[0],means_bounds,Random_State=random_state)
    >>>         samples = sampler.GetSamples(burnin=burn_samples, keep=keep_samples, p_width=p_width)

    >>>         # Initialize the comparison
    >>>         comparison = Comparison(distribution[0], integration_bounds)

    >>>         # analyze the Metropolis sampler performance
    >>>         comparison.Metropolis_Comparison(samples, num_bins)

    >>>         # get error for Metropolis Histogram
    >>>         print(distribution[1], " using MH with p width = ", p_width, " has error: ",
    >>>               comparison.get_Metropolis_integration_error()) 
    >>>         storage_for_plotting.append((comparison, distribution[1], p_width))

    >>>     # Initialize the Chebyshev Interpolant
    >>>     chebyshev = ChebyshevInterpolant(distribution[0],integration_bounds,num_interpolating_points)

    >>>     # analyze the Chebyshev Interpolant performance
    >>>     comparison.Chebyshev_Comparison(chebyshev)

    >>>     # get error for Chebyshev Interpolation
    >>>     print(distribution[1], " using CI with nodes = ", num_interpolating_points, " has error: ",
    >>>         comparison.get_Chebyshev_integration_error()) 
    >>>     storage_for_plotting.append((comparison, distribution[1]))


    >>> # plot all the graphs
    >>> i = 0 # index variable for the loop
    >>> while i < len(storage_for_plotting):
    >>>     # alternate between the 2 methods carefully since storage_for_plotting's elements have differing lengths
    >>>     for p_width in proposal_widths:
    >>>         storage_for_plotting[i][0].plot_MCMC_histogram_estimation(storage_for_plotting[i][1], storage_for_plotting[i][2]) # get histogram
    >>>         i += 1
    >>>     storage_for_plotting[i][0].plot_Chebyshev_Interpolation_estimation(storage_for_plotting[i][1], num_interpolating_points)
    >>>     i+=1



    Attributes
    ----------
    distribution : GMMDataset object
        target distribution used to compare the 2 methods    
    integration_bounds: int
        specifies bounds that indicate where the majority of the mass in the distribution lies.
        integration occurs on these bounds and the visualizations are also truncated here
    metropolis_samples: list of ints
        samples provided from running metropolis algorithm on target distribution
    interpolant: ChebyshevInterpolant object
        interpolant constructed from the target distribution
    '''
    def __init__(self, distribution, integration_bounds):
        self.distribution = distribution
        self.integration_bounds = integration_bounds


    ''' initialize variables for analysis of the metropolis histogram method

        Args
        ----
        metropolis_samples: samples drawn from the given distribution using Metropolis method
        num_bins: number of bins in the histogram

        Returns
        -------
        None. Defines attributes
    '''
    def Metropolis_Comparison(self, metropolis_samples, num_bins):
        hist, bins = np.histogram(metropolis_samples, bins=num_bins, range=self.integration_bounds)
        self.metropolis_samples = metropolis_samples
        self.num_bins = num_bins
        self.hist = hist
        self.bins = bins


    ''' initialize variables for analysis of the Chebyshev interpolation method

        Args
        ----
        interpolant: chebyshev interpolant of the distribution

        Returns
        -------
        None. Defines attributes
    '''
    def Chebyshev_Comparison(self, interpolant):
        interpolant.normalize_interpolant()
        self.interpolant = interpolant
        self.x = sympy.Symbol('x')


    ''' defines a function for numerical integration of the difference between interpolant and distribution. Evaluates 
        absolute difference at a point.

        Args
        ----
        x_0: the value at which to compute the absolute difference

        Returns
        -------
        absolute difference between Interpolant and distribution
    '''
    def Chebyshev_difference_for_integrating(self, x_0):
        x_mass = self.interpolant.get_interpolant().subs(self.x, x_0) # probability of x_0 from the interpolant
        return abs(self.distribution.Evaluate(x_0) - x_mass)
        

    ''' assists numerical integration of the difference between histogram and distribution. Evaluates 
        absolute difference at a point.

        Args
        ----
        x: the value at which to compute the absolute difference

        Returns
        -------
        absolute difference between histogram and distribution
    '''
    def histogram_difference_for_integrating(self, x):
        if x >= self.integration_bounds[1] or x <= self.integration_bounds[0]: # accounts for integration at the endpoints
            return 0
        
        bin_width = (self.integration_bounds[1] - self.integration_bounds[0]) / self.num_bins
        x_bin = int(np.floor((x - self.integration_bounds[0]) / bin_width))

        x_mass = (self.hist[x_bin] / len(self.metropolis_samples)) / bin_width # normalize and preserve the ratios 
        return abs(self.distribution.Evaluate(x) - x_mass)
        

    ''' calculates numerical integration of the difference between histogram and distribution.

        Args
        ----
        x_0: the value at which to compute the absolute difference

        Returns
        -------
        l1 difference between histogram and distribution
    '''
    def get_Metropolis_integration_error(self):
        error = 0
        curr_error = 0
        instability = 0
        for bin in range(self.num_bins):
            curr_error, cur_unstable = integrate.quad(self.histogram_difference_for_integrating,self.bins[bin],self.bins[bin+1])
            error += curr_error
            instability += cur_unstable
        # print("instability is", instability) # optional. Tells how confident the integration was
        return error
    

    ''' calculates numerical integration of the difference between interpolant and distribution.

        Args
        ----
        x_0: the value at which to compute the absolute difference

        Returns
        -------
        l1 difference between interpolant and distribution
    '''
    def get_Chebyshev_integration_error(self):
        error = 0
        instability = 0
        num_intervals = 40
        intervals = np.linspace(self.integration_bounds[0],self.integration_bounds[1],num_intervals)
        for interval in range(num_intervals - 1):
            curr_error, cur_unstable = integrate.quad(self.Chebyshev_difference_for_integrating,intervals[interval],intervals[interval+1])
            error += curr_error
            instability += cur_unstable
        # print("instability is", instability) # optional. Tells how confident the integration was
        return error
    

    ''' plots histogram estimate for distribution

        Args
        ----
        distribution_type: string 
            type of distribution (gaussian eclectic, gaps, concentrated)
        proposal_width: int
            hyperparameter for the Metropolis model.
            helps to identify what is being compared

        Returns
        -------
        None
    '''
    def plot_MCMC_histogram_estimation(self, distribution_type, proposal_width):
        print("histogram of ", distribution_type, " using ", proposal_width, " proposal width")
        plt.hist(self.metropolis_samples, density=True, bins=self.num_bins, range=(self.integration_bounds[0], self.integration_bounds[1]))
        plt.show()


    ''' plots histogram estimate for distribution

        Args
        ----
        distribution_type: string 
            type of distribution (gaussian eclectic, gaps, concentrated)
        num_nodes: int
            hyperparameter for the Interpolation model.
            helps to identify what is being compared

        Returns
        -------
        None
    '''
    def plot_Chebyshev_Interpolation_estimation(self, distribution_type, num_nodes):
        print("interpolation on ", distribution_type, " using ", num_nodes, " nodes")
        sympy.plot(self.interpolant.get_interpolant(), (self.x,self.integration_bounds[0],self.integration_bounds[1]))