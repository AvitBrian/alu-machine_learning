#!/usr/bin/env python3
"""
This function calculates the cofactor of a matrix.
"""
det = __import__('determinant').determinant


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix.

    Args:
        matrix (list): A list of lists representing the input matrix.
    Returns:
        list: The cofactor matrix of the input matrix.
    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square matrix or is empty.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    cofactor_matrix = []

    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sign = (-1) ** (i + j)
            sub_matrix = [row[:j] + row[j + 1:] for row in (
                    matrix[:i] + matrix[i + 1:])]
            cofactor_row.append(sign * det(sub_matrix))
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
