#!/usr/bin/env python3
"""
Module for transposing matrices using NumPy.

This module provides a function, np_transpose, for transposing a matrix using
the NumPy library.

Example:
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    transposed_matrix = np_transpose(matrix)
    print("Original Matrix:")
    print(matrix)
    print("Transposed Matrix:")
    print(transposed_matrix)
"""

import numpy as np


def np_transpose(matrix):
    """
    Transpose a matrix using NumPy.

    Args:
        matrix (numpy.ndarray): The matrix to be transposed.

    Returns:
        numpy.ndarray: The transposed matrix.

    Example:
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        transposed_matrix = np_transpose(matrix)
        print("Original Matrix:")
        print(matrix)
        print("Transposed Matrix:")
        print(transposed_matrix)
    """
    return matrix.T
