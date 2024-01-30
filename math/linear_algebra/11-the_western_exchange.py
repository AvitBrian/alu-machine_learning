#!/usr/bin/env python3
"""
Module for transposing matrices.

This module provides a function, matrix_transpose,
for transposing a matrix.

"""


def np_transpose(matrix):
    """
    Transpose a matrix.

    Args:
        matrix (list): The matrix to be transposed.

    Returns:
        list: The transposed matrix.

    Example:
        matrix = [[1, 2, 3], [4, 5, 6]]
        transposed_matrix = np_transpose(matrix)
    """
    return matrix.transpose()
