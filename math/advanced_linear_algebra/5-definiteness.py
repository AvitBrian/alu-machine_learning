"""
    This function calculates the definiteness of a matrix.
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.
    Args:
        matrix (numpy.ndarray): A numpy array.
    Returns:
        str or None: The definiteness of the matrix.
    Raises:
        TypeError: If matrix is not a numpy.ndarray.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.size == 0 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues, _ = np.linalg.eig(matrix)
    positives = sum(eigenvalues > 0)
    negatives = sum(eigenvalues < 0)
    zeroes = sum(np.isclose(eigenvalues, 0))
    if positives == matrix.shape[0]:
        return "Positive definite"
    elif positives == matrix.shape[0] - zeroes:
        return "Positive semi-definite"
    elif negatives == matrix.shape[0]:
        return "Negative definite"
    elif negatives == matrix.shape[0] - zeroes:
        return "Negative semi-definite"
    else:
        return "Indefinite"
