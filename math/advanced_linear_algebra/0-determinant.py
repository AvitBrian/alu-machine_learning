#!/usr/bin/env python3
"""
This function calculates the determinant of a matrix.
"""


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

    if matrix == []:
        return 1

    if num_rows == 1:
        return matrix[0][0]

    if num_rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    if num_rows != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    det = 0
    for j in range(num_rows):
        det += ((-1) ** j) * matrix[0][j] * determinant(
            [row[:j] + row[j + 1:] for row in matrix[1:]])
    return det
