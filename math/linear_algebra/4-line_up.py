#!/usr/bin/env python3
def add_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return None
    pairs = zip(arr1, arr2)
    sum = []
    for a, b in pairs:
        sum.append(a+b)
    return sum
