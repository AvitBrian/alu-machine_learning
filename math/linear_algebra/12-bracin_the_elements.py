#!/usr/bin/env python3
"""
Module for element-wise matrix operations.

This module provides a function, np_elementwise, 
for performing element-wise
addition, subtraction, multiplication, and division on two matrices.

Example:
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    add, sub, mul, div = np_elementwise(mat1, mat2)
    print("Addition:", add)
    print("Subtraction:", sub)
    print("Multiplication:", mul)
    print("Division:", div)
"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise matrix operations.

    Args:
        mat1 (list): The first matrix.
        mat2 (list): The second matrix.

    Returns:
        tuple: A tuple containing the element-wise sum, difference, product,
            and quotient, respectively.

    Example:
        mat1 = [[1, 2], [3, 4]]
        mat2 = [[5, 6], [7, 8]]
        add, sub, mul, div = np_elementwise(mat1, mat2)
        print("Addition:", add)
        print("Subtraction:", sub)
        print("Multiplication:", mul)
        print("Division:", div)
    """
    addition = [[a + b for a, b in zip(row1, row2)]
                for row1, row2 in zip(mat1, mat2)]
    subtraction = [[a - b for a, b in zip(row1, row2)]
                   for row1, row2 in zip(mat1, mat2)]
    multiplication = [
        [a * b for a, b in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
    division = [[a / b for a, b in zip(row1, row2)]
                for row1, row2 in zip(mat1, mat2)]
    return addition, subtraction, multiplication, division
