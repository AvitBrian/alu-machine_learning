#!/usr/bin/env python3
"""
this function Calculates the intersection of obtaining this data
with the various hypothetical probabilities.
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining this data with
    the various hypothetical probabilities.

    Parameters:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): 1D array containing the various hypothetical
        probabilities of developing severe side effects.
        Pr (numpy.ndarray): 1D array containing the prior beliefs of P.

    Returns:
        numpy.ndarray: Intersection of obtaining x and n
        with each probability in P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1 or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not all(0 <= p <= 1 for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not all(0 <= pr <= 1 for pr in Pr):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(n - x) * factorial(x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))

    intersection = likelihood * Pr

    return intersection
