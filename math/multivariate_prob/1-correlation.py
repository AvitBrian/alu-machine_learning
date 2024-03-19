#!/usr/bin/env python3

import numpy as np

def correlation(C):
  """
  Calculates a correlation matrix from a covariance matrix.

  Args:
      C: A 2D numpy.ndarray of shape (d, d) containing the covariance matrix.

  Returns:
      A numpy.ndarray of shape (d, d) containing the correlation matrix.

  Raises:
      TypeError: If C is not a numpy.ndarray.
      ValueError: If C is not a 2D square matrix.
  """

  if not isinstance(C, np.ndarray):
    raise TypeError("C must be a numpy.ndarray")


  if C.ndim != 2 or C.shape[0] != C.shape[1]:
    raise ValueError("C must be a 2D square matrix")
  diagonal = np.diag(np.ones(C.shape[0]))
  std_dev = np.sqrt(np.add(C, diagonal))  # Standard deviations along the diagonal

  correlation_matrix = C / std_dev[:, np.newaxis] / std_dev[np.newaxis, :]

  correlation_matrix[np.isnan(correlation_matrix)] = 0

  return correlation_matrix