#!/usr/bin/env python3
"""
Module for element-wise matrix operations using NumPy.

This module provides a function, np_elementwise, for performing element-wise
addition, subtraction, multiplication, and division on two matrices using the
NumPy library.

Example:
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    add, sub, mul, div = np_elementwise(mat1, mat2)
    print("Addition:", add)
    print("Subtraction:", sub)
    print("Multiplication:", mul)
    print("Division:", div)
"""

import numpy as np


def np_elementwise(mat1, mat2):
    """
    Perform element-wise matrix operations using NumPy.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        tuple: A tuple containing the element-wise sum, difference, product,
            and quotient, respectively.

    Example:
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[5, 6], [7, 8]])
        add, sub, mul, div = np_elementwise(mat1, mat2)
        print("Addition:", add)
        print("Subtraction:", sub)
        print("Multiplication:", mul)
        print("Division:", div)
    """
    addition = mat1 + mat2
    subtraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return addition, subtraction, multiplication, division
