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

    def determinant(mat):
        """
        Calculates the determinant of a matrix.
        """
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        det = 0
        for j in range(num_rows):
            det += ((-1) ** j) * matrix[0][j] * determinant(
                [row[:j] + row[j + 1:] for row in matrix[1:]])
        return det

    minor_mat = []
    for i in range(num_rows):
        minor_row = []
        for j in range(num_rows):
            submatrix = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
            minor_row.append(determinant(submatrix))
        minor_mat.append(minor_row)
    return minor_mat
