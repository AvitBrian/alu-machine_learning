#!/usr/bin/env python3
"""
Module for element-wise matrix operations.

This module provides a function, np_elementwise,
for performing element-wise
addition, subtraction, multiplication, and division
on two matrices.

"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise matrix operations.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.

    Returns:
        tuple: A tuple containing the element-wise sum,
        difference, product, and quotient, respectively.

    Example:
        mat1 = [[1, 2], [3, 4]]
        mat2 = [[5, 6], [7, 8]]
        add, sub, mul, div = np_elementwise(mat1, mat2)
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
