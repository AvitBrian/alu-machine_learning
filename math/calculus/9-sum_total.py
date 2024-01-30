#!/usr/bin/env python3
'''
    This function
    calculates the sum of the squares of the first n natural numbers
'''


def summation_i_squared(n):
    '''
    Returns the sum of the squares of the first n natural numbers
    '''
    if type(n) is not int or n < 1:
        return None
    sigma_sum = (n * (n + 1) * ((2 * n) + 1)) / 6
    return int(sigma_sum)
