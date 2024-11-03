#!/usr/bin/env python3
'''Performing the Baum-Welch algorithm for a hidden markov model'''

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    '''Function that performs the Baum-Welch algorithm for a hidden markov model

    Parameters:
    Observations: numpy.ndarray of shape (T,) containing the index of the observation
    Transition: numpy.ndarray of shape (M, M) containing the initialized transition probabilities
    Emission: numpy.ndarray of shape (M, N) containing the initialized emission probabilities
    Initial: numpy.ndarray of shape (M, 1) containing the initialized starting probabilities
    iterations: number of times expectation-maximization should be performed

    Returns: the converged Transition, Emission, or None, None on failure
    '''
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if Observations.shape[0] == 0:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if Emission.shape[0] != Transition.shape[0]:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if Initial.shape[0] != Emission.shape[0] or Initial.shape[1] != 1:
        return None, None

    return None, None
