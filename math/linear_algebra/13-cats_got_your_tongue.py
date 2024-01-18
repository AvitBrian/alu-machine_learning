#!/usr/bin/env python3
"""
Module for concatenating matrices using NumPy.

This module provides a function, np_cat, for concatenating matrices along a
specific axis using the NumPy library.

Example:
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    result = np_cat(mat1, mat2, axis=0)
    print(result)
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate matrices along a specific axis using NumPy.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.
        axis (int, optional): The axis along which the matrices will be
            concatenated. Defaults to 0.

    Returns:
        numpy.ndarray: The result of concatenation.

    Raises:
        ValueError: If the matrices are not compatible for concatenation.

    Example:
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[5, 6], [7, 8]])
        result = np_cat(mat1, mat2, axis=0)
        print(result)
    """
    return np.concatenate((mat1, mat2), axis=axis)
