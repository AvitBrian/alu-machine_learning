#!/usr/bin/env python3
"""
Module for adding two arrays element-wise.

This module provides a function to add two arrays element-wise.

Example:
    arr1 = [1, 2, 3, 4]
    arr2 = [5, 6, 7, 8]
    result = add_arrays(arr1, arr2)
    print(result)
"""


def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise.

    Args:
        arr1 (list): The first array.
        arr2 (list): The second array.

    Returns:
        list: The resulting array after element-wise addition.
        None: If arrays have different lengths.
    """
    if len(arr1) != len(arr2):
        return None
    pairs = zip(arr1, arr2)
    sum_result = []
    for a, b in pairs:
        sum_result.append(a + b)
    return sum_result
