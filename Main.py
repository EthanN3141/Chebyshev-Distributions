# Author: Ethan Nanavati
# Last Edited: 6/28/24
# Known Bugs: None


import numpy as np

from Dataset_Management_Class import GMMDataset
from Metropolis_Class import Metropolis
from Chebyshev_Class import ChebyshevInterpolant
from Method_Comparison import Comparison


# set hyperparameters
means_bounds = [-10,10] # the gaussians in the distribution will have means in this range
integration_bounds = [-20,20] # captures at least 99.96% of the mass with means=[-10,10]
num_peaks = 8 # gausian mixture will have this many means
burn_samples = 5000
keep_samples = 5000
num_bins = 3 * int(1 + np.ceil(3.322 * np.log(keep_samples))) # number of bins chosen via Sturge's rule: K = 1 + 3. 322 logN.
num_interpolating_points = 28
proposal_widths = [5] #[1,5,10]
random_state = 55 # which random state to use


# Create the distributions
distribution_GC = GMMDataset(num_peaks,means_bounds, 'GC',Random_State=random_state)
distribution_GE = GMMDataset(num_peaks,means_bounds, 'GE',Random_State=random_state)
distribution_GG = GMMDataset(num_peaks,means_bounds, 'GG',Random_State=random_state)
distributions = [(distribution_GC, "GC"), (distribution_GE, "GE"), (distribution_GG, "GG")] # to be looped through


# performs analysis on the 2 methods
storage_for_plotting = [] # stores info for the graphs at the end
for distribution in distributions:
    for p_width in proposal_widths:  
        # Initialize the Metropolis sampler  
        sampler = Metropolis(distribution[0],means_bounds,Random_State=random_state)
        samples = sampler.GetSamples(burnin=burn_samples, keep=keep_samples, p_width=p_width)

        # Initialize the comparison
        comparison = Comparison(distribution[0], integration_bounds)

        # analyze the Metropolis sampler performance
        comparison.Metropolis_Comparison(samples, num_bins)

        # get error for Metropolis Histogram
        print(distribution[1], " using MH with p width = ", p_width, " has error: ",
               comparison.get_Metropolis_integration_error()) 
        storage_for_plotting.append((comparison, distribution[1], p_width))

    # Initialize the Chebyshev Interpolant
    chebyshev = ChebyshevInterpolant(distribution[0],integration_bounds,num_interpolating_points)

    # analyze the Chebyshev Interpolant performance
    comparison.Chebyshev_Comparison(chebyshev)

    # get error for Chebyshev Interpolation
    print(distribution[1], " using CI with nodes = ", num_interpolating_points, " has error: ",
          comparison.get_Chebyshev_integration_error()) 
    storage_for_plotting.append((comparison, distribution[1]))


# plot all the graphs
i = 0 # index variable for the loop
while i < len(storage_for_plotting):
    # alternate between the 2 methods carefully since storage_for_plotting's elements have differing lengths
    for p_width in proposal_widths:
        storage_for_plotting[i][0].plot_MCMC_histogram_estimation(storage_for_plotting[i][1], storage_for_plotting[i][2]) # get histogram
        i += 1
    storage_for_plotting[i][0].plot_Chebyshev_Interpolation_estimation(storage_for_plotting[i][1], num_interpolating_points)
    i+=1
    