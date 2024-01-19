#!/usr/bin/env python3
"""
Module for performing matrix multiplication.

This module provides a function to perform matrix multiplication
on two matrices.
The matrices must be 2D and have compatible dimensions for multiplication.

Example:
    mat1 = [[1, 2],
            [3, 4],
            [5, 6]]
    mat2 = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
    result = mat_mul(mat1, mat2)
    print(result)
"""


def mat_mul(mat1, mat2):
    """
    Perform matrix multiplication on two matrices.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.

    Returns:
        list or None: The result of matrix multiplication
        or None if matrices are incompatible.
    """
    if len(mat1[0]) != len(mat2):
        return None

    new_mat = []
    for i in range(len(mat1)):
        temp_row = []
        for j in range(len(mat2[0])):
            sum = 0
            for k in range(len(mat2)):
                sum += mat1[i][k] * mat2[k][j]
            temp_row.append(sum)
        new_mat.append(temp_row)

    return new_mat
