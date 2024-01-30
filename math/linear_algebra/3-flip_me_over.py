#!/usr/bin/env python3
"""
Module for transposing a 2D matrix.

This module provides a function to transpose a 2D matrix.

Example:
    mat1 = [[1, 2], [3, 4]]
    transposed_mat = matrix_transpose(mat1)
    print(transposed_mat)
"""


def matrix_transpose(matrix):
    """
    Transpose a 2D matrix.

    Args:
        matrix (list): The input 2D matrix.

    Returns:
        list: The transposed matrix.
    """
    transposed_matrix = []
    for row in zip(*matrix):
        transposed_matrix.append(list(row))
    return transposed_matrix
