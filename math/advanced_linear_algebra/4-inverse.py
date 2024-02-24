#!/usr/bin/env python3
"""
    This function calculates the inverse of a matrix.
"""


def inverse(matrix):
    """
    Calculates the inverse of a matrix.
    Args:
        matrix (list): A list of lists representing the input matrix.
    Returns:
        list: The inverse of the input matrix,
         or None if matrix is singular.
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
    det = __import__('determinant').determinant(matrix)
    if det == 0:
        return None

    adj = __import__('3-adjugate').adjugate(matrix)

    inverse_matrix = [[adj[i][j] / det
                       for j in range(n)] for i in range(n)]

    return inverse_matrix
