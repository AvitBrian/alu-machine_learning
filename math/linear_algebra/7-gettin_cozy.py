#!/usr/bin/env python3
"""
Module for concatenating 2D matrices along a specific axis.

This module provides a function to concatenate two matrices
along a specified axis.
The matrices must have compatible dimensions for concatenation.

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
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
