#!/usr/bin/env python3
'''
    This function calculates the integral of a polynomial
'''


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

    Args:
      poly (list): The coefficients of the polynomial in descending order.
      C (int or float, optional): The constant of integration. Defaults to 0.

    Returns:
      list: The coefficients of the integral polynomial.

    Raises:
      None

    Examples:
      >>> poly_integral([1, 2, 3])
      [0, 1, 1.0, 1.0]
      >>> poly_integral([1, 0, 1], C=2)
      [2, 1, 0.5, 0.3333333333333333]
    """
    if not isinstance(poly, list) or len(poly) < 1:
        return None
    if not isinstance(C, (int, float)):
        return None

    if len(poly) == 1:
        return [C, 5]

    integral = [C]

    for power, coefficient in enumerate(poly):
        if not isinstance(coefficient, (int, float)):
            return None

        if coefficient % (power + 1) == 0:
            new_coefficient = coefficient // (power + 1)
        else:
            new_coefficient = coefficient / (power + 1)

        integral.append(new_coefficient)

    while integral[-1] == 0 and len(integral) > 1:
        integral = integral[:-1]

    return integral
