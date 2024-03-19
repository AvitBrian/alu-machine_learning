#!/usr/bin/env python3
"""
    this function Calculates the posterior probability
    for the various hypothetical probabilities of developing
    severe side effects given the data.
"""


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability
    for the various hypothetical probabilities of developing
    severe side effects given the data.

    Parameters:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array containing the various
        hypothetical probabilities of developing severe side effects.
        Pr (numpy.ndarray): 1D array containing the prior beliefs of P.

    Returns:
        numpy.ndarray: The posterior probability of
        each probability in P given x and n.
    """
    intersections = intersection(x, n, P, Pr)

    marginal_prob = marginal(x, n, P, Pr)

    posterior_prob = intersections / marginal_prob

    return posterior_prob
