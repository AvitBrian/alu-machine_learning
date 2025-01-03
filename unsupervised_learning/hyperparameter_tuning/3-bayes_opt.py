#!/usr/bin/env python3
'''
This module initializes Bayesian Optimization
'''
import numpy as np
GP = __import__('2-gp').GaussianProcess

class BayesianOptimization:
    '''
    Performs Bayesian optimization on a Gaussian process.
    '''

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        '''
        Initializes Bayesian Optimization.
        '''
        MIN, MAX = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(MIN, MAX, num=ac_samples)[..., np.newaxis]
        self.xsi = xsi
        self.minimize = minimize
