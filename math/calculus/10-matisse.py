#!/usr/bin/env python3
'''
    This function calculates the derivative of a polynomial
'''


def poly_derivative(poly):
    '''
    Returns the derivative of a polynomial
    '''
    if type(poly) is not list or not\
          all(isinstance(coef, (int, float)) for coef in poly):
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power, coef in enumerate(poly[1:], start=1):
        derivative.append(coef * power)

    return derivative
