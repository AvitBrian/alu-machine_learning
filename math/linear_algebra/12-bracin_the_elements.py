#!/usr/bin/env python3
"""
Module for element-wise matrix operations.

This module provides a function, np_elementwise, 
for performing element-wise
addition, subtraction, multiplication, and division
on two matrices.

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
        tuple: A tuple containing the element-wise sum, 
        difference, product, and quotient, respectively.

    Example:
        mat1 = [[1, 2], [3, 4]]
        mat2 = [[5, 6], [7, 8]]
        add, sub, mul, div = np_elementwise(mat1, mat2)
        print("Addition:", add)
        print("Subtraction:", sub)
        print("Multiplication:", mul)
        print("Division:", div)
    """
    addition = list(map(lambda x, y: list(
        map(lambda a, b: a + b, x, y)), mat1, mat2))
    subtraction = list(map(lambda x, y: list(
        map(lambda a, b: a - b, x, y)), mat1, mat2))
    multiplication = list(map(lambda x, y: list(
        map(lambda a, b: a * b, x, y)), mat1, mat2))
    division = list(map(lambda x, y: list(
        map(lambda a, b: a / b, x, y)), mat1, mat2))
    return addition, subtraction, multiplication, division
