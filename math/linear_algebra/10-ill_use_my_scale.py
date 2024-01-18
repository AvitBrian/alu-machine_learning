#!/usr/bin/env python3
"""
Module for calculating the shape of a NumPy array.

This module provides a function, np_shape, for calculating the shape of a NumPy
array.

Example:
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    shape_result = np_shape(matrix)
    print("Matrix Shape:")
    print(shape_result)
"""

import numpy as np


def np_shape(matrix):
    """
    Calculate the shape of a NumPy array.

    Args:
        matrix (numpy.ndarray): The NumPy array.

    Returns:
        Tuple: A tuple representing the shape of the array.

    Example:
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        shape_result = np_shape(matrix)
        print("Matrix Shape:")
        print(shape_result)
    """
    return matrix.shape
