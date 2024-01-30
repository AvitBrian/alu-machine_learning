#!/usr/bin/env python3
'''
    This function calculates the derivative of a polynomial
'''


def poly_derivative(poly):
    '''
    Returns the derivative of a polynomial
    '''
    if not isinstance(poly, list):
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power, coef in enumerate(poly[1:], start=1):
        derivative.append(coef * power)

    if not derivative:
        return None

    return derivative
