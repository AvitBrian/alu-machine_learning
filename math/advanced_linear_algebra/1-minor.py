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

    num_rows = len(matrix)
    if num_rows == 0 or any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_mat = []
    for i in range(num_rows):
        minor_row = []
        for j in range(num_rows):
            minor_row.append(determinant([row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]))
        minor_mat.append(minor_row)
    return minor_mat
