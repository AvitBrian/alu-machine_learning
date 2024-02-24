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
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    minor_matrix = []

    for i in range(n):
        minor_row = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j + 1:] for row in (
                    matrix[:i] + matrix[i + 1:])]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix (list): A list of lists whose determinant should be calculated.
    Returns:
        int: The determinant of the matrix.
    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square matrix.
    """

    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0

    if num_rows != num_cols:
        raise ValueError("matrix must be a non-empty square matrix")

    if not matrix:
        return 1

    if num_rows == 1:
        return matrix[0][0]

    if num_rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(num_rows):
        det += ((-1) ** j) * matrix[0][j] * determinant(
            [row[:j] + row[j + 1:] for row in matrix[1:]])
    return det
