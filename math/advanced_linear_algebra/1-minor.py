#!/usr/bin/env python3
"""
    This function Calculates the minor matrix of a square matrix.
"""


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.

    Args:
        matrix (list): A list of lists representing the input matrix.
    Returns:
        list: The minor matrix of the input matrix.
    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square matrix or is empty.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    minor_matrix = []

    for i in range(n):
        minor_row = []
        for j in range(n):
            minor_row.append(matrix[(i + 1) % n][(j + 1) % n] *
                             matrix[(i + 2) % n][(j + 2) % n] -
                             matrix[(i + 1) % n][(j + 2) % n] *
                             matrix[(i + 2) % n][(j + 1) % n])
        minor_matrix.append(minor_row)

    return minor_matrix


print(minor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
