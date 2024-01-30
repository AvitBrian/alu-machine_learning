#!/usr/bin/env python3
'''
    This function calculates the integral of a polynomial
'''


def poly_integral(poly, C=0):
    """
    Returns the integral of a polynomial
    """
    if type(poly) is not list or len(poly) < 1:
        return None
    if type(C) is not int and type(C) is not float:
        return None
    if len(poly) == 1:
        return [C]
    integral = [C]
    for i in range(len(poly)):
        if type(poly[i]) is not int and type(poly[i]) is not float:
            return None
        integral.append(poly[i] / (i + 1))
    integral = [int(i) if i % 1 == 0 else i for i in integral]
    return integral
  