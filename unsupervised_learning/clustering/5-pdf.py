#!/usr/bin/env python3
"""
Module for calculating the probability density function of a Gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0] or S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape

    try:
        x_minus_mean = X - m

        inv_S = np.linalg.inv(S)
        exponent = np.sum(np.matmul(x_minus_mean, inv_S) * x_minus_mean, axis=1)
        coefficient = 1 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(S)))

        P = coefficient * np.exp(-0.5 * exponent)
        P = np.maximum(P, 1e-300)

        return P
    except:
        return None
