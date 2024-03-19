#!/usr/bin/env python3
"""
Calculates the marginal probability of obtaining the data.
"""


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
    intersection = intersection(x, n, P, Pr)
    marginal_prob = np.sum(intersection)

    return marginal_prob
