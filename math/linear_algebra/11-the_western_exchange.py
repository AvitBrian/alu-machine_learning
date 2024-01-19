#!/usr/bin/env python3
"""
Module for transposing matrices.

This module provides a function, matrix_transpose,
for transposing a matrix.

Example:
    matrix = [[1, 2, 3], [4, 5, 6]]
    transposed_matrix = matrix_transpose(matrix)
    print("Original Matrix:")
    print(matrix)
    print("Transposed Matrix:")
    print(transposed_matrix)
"""


def matrix_transpose(matrix):
    """
    Transpose a matrix.

    Args:
        matrix (list): The matrix to be transposed.

    Returns:
        list: The transposed matrix.

    Example:
        matrix = [[1, 2, 3], [4, 5, 6]]
        transposed_matrix = matrix_transpose(matrix)
        print("Original Matrix:")
        print(matrix)
        print("Transposed Matrix:")
        print(transposed_matrix)
    """
    transposed_matrix = []
    for row in zip(*matrix):
        transposed_matrix.append(list(row))
    return transposed_matrix
