#!/usr/bin/env python3
"""
Calculates the marginal probability of obtaining the data.
"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data.

    Parameters:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array containing the various hypothetical probabilities
        of patients developing severe side effects.
        Pr (numpy.ndarray): 1D array containing the prior beliefs about P.

    Returns:
        float: The marginal probability of obtaining x and n.
    """
    for value in range(P.shape[0]):
        if P[value] > 1 or P[value] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[value] > 1 or Pr[value] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")

    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(n - x) * factorial(x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))

    intersection = likelihood * Pr

    marginal_prob = np.sum(intersection)

    return marginal_prob
