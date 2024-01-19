#!/usr/bin/env python3
"""
Module for concatenating 2D matrices along a specific axis.

This module provides a function to concatenate two matrices along a specified axis.
The matrices must have compatible dimensions for concatenation.

Example:
    mat1 = [[1, 2],
            [3, 4]]
    mat2 = [[5, 6],
            [7, 8]]
    result = cat_matrices2D(mat1, mat2, axis=0)
    print(result)
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specific axis.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.
        axis (int, optional): The axis along which to concatenate.
        Defaults to 0.

    Returns:
        list or None: The concatenated matrix
        or None if matrices are incompatible.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        result_matrix = [row.copy() for row in mat1]
        result_matrix.extend([row.copy() for row in mat2])

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        result_matrix = [mat1[i].copy() + mat2[i] for i in range(len(mat1))]

    return result_matrix


# Example Usage:
mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
result = cat_matrices2D(mat1, mat2, axis=0)
print(result)
