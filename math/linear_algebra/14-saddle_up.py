#!/usr/bin/env python3
import numpy as np

def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication using NumPy.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The result of matrix multiplication.
        
    Raises:
        ValueError: If the matrices are not compatible for multiplication.

    Example:
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[5, 6], [7, 8]])
        result = np_matmul(mat1, mat2)
        print(result)
    """
    return np.matmul(mat1, mat2)
