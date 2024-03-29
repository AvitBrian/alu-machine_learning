#!/usr/bin/env python3
"""
Module for calculating the shape of a NumPy array.

This module provides a function, np_shape,
for calculating the shape of a NumPy
array.

"""


def np_shape(matrix):
    """
    Calculate the shape of a NumPy array.

    Args:
        matrix (list or nested list): The array.

    Returns:
        Tuple: A tuple representing the shape of the array.

    Example:
        matrix = [[1, 2, 3], [4, 5, 6]]

    """
    return tuple(matrix.shape)
