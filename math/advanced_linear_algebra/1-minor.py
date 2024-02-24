#!/usr/bin/env python3
"""
    This function Calculates the minor matrix of a square matrix.
"""
det = __import__('math/advanced_linear_algebra/0-determinant').determinant


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
            sub_matrix = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
            minor_row.append(det(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix
