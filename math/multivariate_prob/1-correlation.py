#!/usr/bin/env python3
"""
  this function Calculates a correlation matrix
  from a covariance matrix.
"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix.

    Args:
        C: A 2D numpy.ndarray of shape (d, d) containing the covariance matrix.

    Returns:
        A numpy.ndarray of shape (d, d) containing the correlation matrix.

    Raises:
        TypeError: If C is not a numpy.ndarray.
        ValueError: If C is not a 2D square matrix.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    diagonal = np.sqrt(np.diag(C))
    correlation_matrix = C / np.outer(diagonal, diagonal)
    return correlation_matrix
