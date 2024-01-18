#!/usr/bin/env python3
"""
Module for concatenating two arrays.

This module provides a function to concatenate two arrays.

Example:
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]
    result = cat_arrays(arr1, arr2)
    print(result)
"""


def cat_arrays(arr1, arr2):
    """
    Concatenate two arrays.

    Args:
        arr1 (list): The first array.
        arr2 (list): The second array.

    Returns:
        list: The concatenated array.
    """
    cat_array = []
    for element in arr1:
        cat_array.append(element)
    for element in arr2:
        cat_array.append(element)
    return cat_array
