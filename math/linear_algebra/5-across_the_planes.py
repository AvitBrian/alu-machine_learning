#!/usr/bin/env python3
"""
Module for adding two 2D matrices element-wise.

This module provides a function to add two 2D matrices element-wise.

Example:
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    result = add_matrices2D(mat1, mat2)
    print(result)
"""


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-wise.

    Args:
        mat1 (list): The first 2D matrix.
        mat2 (list): The second 2D matrix.

    Returns:
        list: The resulting 2D matrix after element-wise addition.
        None: If matrices have different shapes.
    """
    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
    sum_matrix = []
    for row1, row2 in zip(mat1, mat2):
        sum_row = []
        for a, b in zip(row1, row2):
            sum_row.append(a + b)
        sum_matrix.append(sum_row)

    return sum_matrix
