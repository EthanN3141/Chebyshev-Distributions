# Author: Ethan Nanavati
# Last Edited: 6/28/24
# Known Bugs: None

import numpy as np
import sympy

from Dataset_Management_Class import GMMDataset # remove after testing



class ChebyshevInterpolant(object):
    '''    A class that allows you to construct and interact with a chebyshev interpolant of a function

    Arguments: 
    -------------------
    function: function to sample from   *actually it is a distribution from my class
        must be evaluatable as a single variable function using function(x)
        it is assumed that the function has the vast majority of its mass concentrated between the bounds specified by 'bounds'
        In the intended use, this function will be a probability distribution
    num_nodes: number of nodes for interpolation
    bounds: specifies the range of values that we assume f is defined on
        input as a list of 2 numbers [lower, upper]
        note that this is required both to define the Chebyshev nodes and to integrate the interpolant

    Example Usage:
    >>> import matplotlib.pyplot as plt
    >>> import sympy

    # Create the target distribution
    >>> dataset = GMMDataset(2,[-10,10],"GG")
    >>> dataset.Plot_Distribution()

    # Create the interpolant
    >>> Cheb = ChebyshevInterpolant(dataset,[-20,20],16)
    >>> Cheb.normalize_interpolant()

    # Get the interpolant
    >>> interpolant = Cheb.get_interpolant()

    # visualize the interpolant
    >>> sympy.plot(interpolant, (x,-20,20))
    -------------------    
    '''
    x = sympy.Symbol('x')


    ''' Draw sample z_prime from the proposal distribution centered at center

        Args
        ----
        function : GMMDataset
            distribution that we will interpolate
            draws from a scaled version of the distribution provided by the object
        bounds : [int,int]
            provides the upper and lower bounds for the integration
            note that this is required for interpolation, but a distribution may have non-compact support
        num_nodes: int
            number of nodes used for interpolation

        Returns
        -------
        None. Sets attributes
    '''
    def __init__(self, function, bounds, num_nodes=16):
        self.lower = bounds[0]
        self.upper = bounds[1]
        self.num_nodes = num_nodes
        self.function = function
        self._CalcPoints()
        self._ConstructPolynomial()


    ''' Calculates the chebyshev points that will be used in interpolation based on num_nodes and bounds

        Args
        ----

        Returns
        -------
        None. 
    '''
    # if this were private, that would be lit
    def _CalcPoints(self):
        Chebyshev_Points = []

        # interval is [a,b]
        a = self.lower
        b = self.upper
        n = self.num_nodes

        # Calculate each of the Chebyshev points
        for i in range(self.num_nodes):
            curr_point = (a+b)/2 + (b-a)/2 *np.cos(np.pi*(2*i+1)/(2*n))
            Chebyshev_Points.append(curr_point)

        self.points = np.array(Chebyshev_Points) # np because indexing is fun


    ''' construct the chebyshev interpolant at the chebyshev points

        Args
        ----
        none

        Returns
        -------
        None. Defines the interpolant attribute
    '''
    # perform lagrange interpolation
    def _ConstructPolynomial(self):
        x_points = self.points

        polynomial = 0 # update this by adding basis functions as it loops
        for i in range(self.num_nodes):
            li_denom = 1
            li_numer = 1

            
            for j in range(self.num_nodes):
                if j != i:
                    li_denom *= (x_points[i] - x_points[j]) # computes the denominator for li the ith lagrange function
                    li_numer = sympy.prod([li_numer, self.x - x_points[j]]) # computes the numerator for li the ith lagrange function
            li_numer = sympy.expand(li_numer)
            yi_li = self.function.Evaluate_Scaled(x_points[i]) * (li_numer / li_denom) #f(xi) * li in the lagrange interpolation
            polynomial += yi_li

        self.interpolant = polynomial

    ''' gets the interpolant attribute

        Args
        ----
        none

        Returns
        -------
        the interpolant in symbolic form
    '''
    def get_interpolant(self):
        return self.interpolant

    ''' computes the definite integral of interpolant on a region to normalize the interpolant

        Args
        ----
        none

        Returns
        -------
        None. normalizes the interpolant attribute
    '''
    # computes the definite integral of interpolant on a region to normalize the interpolant
    def normalize_interpolant(self):
        self.interpolant *= 1 / sympy.integrate(self.interpolant, (self.x, self.lower, self.upper))




# means_bounds = [-10,10]
# integration_bounds = [-20,20] # captures at least 99.96% of the mass with means=[-10,10] and integr=[-20,20]
# num_peaks = 8 # gausian mixture will have this many means
# burn_samples = 5000
# keep_samples = 50000
# num_bins = int(1 + np.ceil(3.322 * np.log(keep_samples))) # number of bins chosen via Sturge's rule: K = 1 + 3. 322 logN. 30 rn

# random_state = 55 # which random state to use

# # Create the Gaussian Eclectic dataset
# distribution_GE = GMMDataset(num_peaks,means_bounds, 'GE',Random_State=random_state)

# interpolant = ChebyshevInterpolant(distribution_GE,integration_bounds,8)

# print(interpolant.ConstructPolynomial())