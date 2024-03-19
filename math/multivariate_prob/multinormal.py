#!/usr/bin/env python3

import numpy as np


class MultiNormal:
  """
  Represents a Multivariate Normal distribution.
  """

  def __init__(self, data):
    """
    Initializes the MultiNormal class.

    Args:
        data: A 2D numpy.ndarray of shape (d, n) containing the data set.
    """

    if not isinstance(data, np.ndarray) or data.ndim != 2:
      raise TypeError("data must be a 2D numpy.ndarray")

    if data.shape[1] < 2:
      raise ValueError("data must contain multiple data points")

    self.mean = np.mean(data, axis=1, keepdims=True)
    centered_data = data - self.mean
    self.cov = np.dot(centered_data, centered_data.T) / (data.shape[1] - 1)

  def pdf(self, x):
    """
    Calculates the probability density function (PDF) at a data point.

    Args:
        x: A numpy.ndarray of shape (d, 1) containing the data point.

    Returns:
        The value of the PDF at the data point.

    Raises:
        TypeError: If x is not a numpy.ndarray.
        ValueError: If x is not of shape (d, 1).
    """

    if not isinstance(x, np.ndarray):
      raise TypeError("x must be a numpy.ndarray")

    if x.shape != (self.cov.shape[0], 1):
      raise ValueError(f"x must have the shape ({self.cov.shape[0]}, 1)")

    centered_x = x - self.mean
    quadratic_term = np.dot(centered_x.T, np.dot(self.cov, centered_x))

    d = self.cov.shape[0]
    constant_term = (2 * np.pi) ** (-d / 2) * np.linalg.det(self.cov) ** (1 / 2)

    return constant_term * np.exp(-0.5 * quadratic_term)