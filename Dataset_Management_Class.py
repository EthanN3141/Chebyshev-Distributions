# Author: Ethan Nanavati
# Last Edited: 6/28/24
# Known Bugs: None
# Notes: I hard coded all the variances. it is important that a correct interval is chosen for the data. 
# Variances may cause a poor fit if not careful. To that end, it is strongly recommended that the interval
# length not be smaller than 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


class GMMDataset(object):
    '''num_peaks is a positive integer
    
    dataset looks like [(mu1,sigma1,pi1), (mu2,sigma2,pi2), ...]
    '''
    def __init__(self, num_peaks, mean_bounds, data_type,Random_State=42):
        self.peaks = num_peaks
        self.lower = mean_bounds[0]
        self.upper = mean_bounds[1]
        self.Random_State = np.random.RandomState(int(Random_State))
        if data_type == "GG":
            self.distribution = self.Gaussian_Gaps()
        elif data_type == "GC":
            self.distribution = self.Gaussian_Concentrated()
        elif data_type == "GE":
            self.distribution = self.Gaussian_Eclectic()
        else:
            raise ValueError("Invalid input. Valid datatypes are 'GG' (Gaussian Gaps), 'GC' (Gausssian Concentrated), 'GE' (Gaussian Eclectic)")
        self.error_constant = .15 # this is the constant that the distribution will be scaled by (used by the metropolis and chebyshev classes)

    # generates means that are far apart from each other (near each bound's endpoint)
    def Gaussian_Gaps(self):
        # self.Random_State.uniform()
        means_low = self.Random_State.uniform(self.lower, self.lower + (self.upper - self.lower)*.1, int(np.floor(self.peaks/2)))
        means_high = self.Random_State.uniform(self.upper - (self.upper - self.lower)*.1, self.upper ,int(np.ceil(self.peaks/2)))

        stdevs = np.sqrt(self.Random_State.uniform(.5,2,self.peaks))

        means = np.concatenate((means_low,means_high))

        # generate random valid weights.
        if self.peaks == 1:
            weights = [1]
        else:
            split = self.Random_State.uniform(0,1,self.peaks - 1)
            split = np.append(split,[0,1])
            split.sort() 
            weights = [split[i+1] - split[i] for i in range(self.peaks)]

        Gaussian_Gaps_Data = list(zip(means,stdevs,weights))
        return Gaussian_Gaps_Data


    # the means are all clustered together. Specifically, they are all in the middle 20% of the provided bounds
    def Gaussian_Concentrated(self):
        means = self.Random_State.uniform((self.lower + self.upper)/2 - (self.upper - self.lower)*.1,
                                  (self.lower + self.upper)/2 + (self.upper - self.lower)*.1, self.peaks)
        
        stdevs = np.sqrt(self.Random_State.uniform(.5,2,self.peaks))

        # generate random valid weights. 
        if self.peaks == 1:
            weights = [1]
        else:
            split = self.Random_State.uniform(0,1,self.peaks - 1)
            split = np.append(split,[0,1])
            split.sort() 
            weights = [split[i+1] - split[i] for i in range(self.peaks)]

        Gaussian_Concentrated_Data = list(zip(means,stdevs,weights))
        return Gaussian_Concentrated_Data


    # means are chosen uniformly randomly through the interval with more variance
    def Gaussian_Eclectic(self):
        means = self.Random_State.uniform(self.lower,self.upper, self.peaks)
        
        stdevs = np.sqrt(self.Random_State.uniform(.2,9,self.peaks)) # there is a larger range of variances here

        # generate random valid weights. 
        if self.peaks == 1:
            weights = [1]
        else:
            split = self.Random_State.uniform(0,1,self.peaks - 1)
            split = np.append(split,[0,1])
            split.sort() 
            weights = [split[i+1] - split[i] for i in range(self.peaks)]

        Gaussian_Eclectic_Data = list(zip(means,stdevs,weights))
        return Gaussian_Eclectic_Data
    
    # if the field of vision is wonky, adjust the x_axis
    def Plot_Distribution(self):
        x_axis = np.arange(abs(self.lower) * -2, abs(self.upper) * 2) 
        y_axis = []
        for x in x_axis:
            y_axis.append(self.Evaluate(x))
        plt.plot(x_axis,y_axis)
        plt.show()
    
    def Get_Distribution(self):
        return self.distribution
    
    def Evaluate(self, x):
        f_of_x = 0
        for gaussian in self.distribution:
            f_of_x += gaussian[2] * scipy.stats.norm.pdf(x, gaussian[0], gaussian[1])
        return f_of_x
    
    # this returns p tilde at the point x
    def Evaluate_Scaled(self,x):
        return self.error_constant * self.Evaluate(x)


# dataset = GMMDataset(8,[-10,10],"GE",Random_State=55)
# dataset.Plot_Distribution()

# dataset = GMMDataset(8,[-10,10],"GG",Random_State=55)
# dataset.Plot_Distribution()

# dataset = GMMDataset(8,[-10,10],"GC",Random_State=55)
# dataset.Plot_Distribution()