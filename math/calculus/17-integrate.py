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

    if i == 0:
      integral.append(poly[i])
    else:
      result = poly[i] / (i + 1)
      new_coefficient = result if result % 1 != 0 else int(result)
      integral.append(new_coefficient)

  while integral[-1] == 0 and len(integral) > 1:
    integral = integral[:-1]

  return integral
