#!/usr/bin/env python3
"""
Module for calculating the shape of a nested list.

This module provides a function to calculate the shape of a nested list,
representing a multi-dimensional matrix.

Example:
    mat = [[1, 2], [3, 4]]
    shape = matrix_shape(mat)
    print(shape)

    mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
    shape2 = matrix_shape(mat2)
    print(shape2)
"""


def matrix_shape(matrix):
    """
    Calculate the shape of a nested list.

    Args:
        matrix (list): The input nested list representing a multi-dimensional matrix.

    Returns:
        list: The shape of the matrix.
    """
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
