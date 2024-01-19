#!/usr/bin/env python3
"""
Module for calculating the shape of a nested list.

This module provides a function to calculate the shape of a nested list,
representing a multi-dimensional matrix.

Example:
    mat = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    shape = matrix_shape(mat)
    print(shape)
"""


def matrix_shape(matrix):
    """
    Calculate the shape of a nested list.

    Args:
        matrix (list): The input nested list
        representing a multi-dimensional matrix.

    Returns:
        list: The shape of the matrix.
    """
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
