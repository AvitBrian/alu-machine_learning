#!/usr/bin/env python3
import numpy as np
"""
Calculate the mean and covariance of a 2D numpy array.

        Parameters:
        X (numpy.ndarray): Input array of shape (n_samples, n_features).

        Returns:
        tuple: A tuple containing the mean and covariance of the input array.
              - mean (numpy.ndarray): Mean of the input array,
              of shape (1, n_features).
              - cov (numpy.ndarray): Covariance matrix of the input array,
              of shape (n_features, n_features).

        Raises:
        TypeError: If X is not a 2D numpy array.
        ValueError: If X contains less than 2 data points.
"""


def mean_cov(X):
    """
        Calculate the mean and covariance of a 2D numpy array.

        Parameters:
        X (numpy.ndarray): Input array of shape (n_samples, n_features).

        Returns:
        tuple: A tuple containing the mean and covariance of the input array.
              - mean (numpy.ndarray): Mean of the input array,
              of shape (1, n_features).
              - cov (numpy.ndarray): Covariance matrix of the input array,
              of shape (n_features, n_features).

        Raises:
        TypeError: If X is not a 2D numpy array.
        ValueError: If X contains less than 2 data points.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    centered_data = X - mean
    cov = np.dot(centered_data.T, centered_data) / (X.shape[0] - 1)

    return mean, cov
