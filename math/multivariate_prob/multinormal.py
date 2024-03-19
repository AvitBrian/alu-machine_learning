#!/usr/bin/env python3
"""
Represents a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes a MultiNormal instance.

        Parameters:
            data (numpy.ndarray):
            Input array of shape (d, n) containing the data set,
            where n is the number of data points and d is the number
            of dimensions in each data point.

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If data contains less than 2 data points.
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
        Calculates the probability density function (PDF)
        at a given data point.

        Parameters:
            x (numpy.ndarray):
            Data point of shape (d, 1) whose PDF should be calculated,
            where d is the number of dimensions of the Multinomial instance.

        Returns:
            float: The value of the PDF at the given data point.

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x is not of shape (d, 1),
            where d is the number of dimensions.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.mean.shape[0], 1):
            raise ValueError("x must have the shape ({}, 1)".format(
                self.mean.shape[0]))

        d = self.mean.shape[0]
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        diff = x - self.mean
        exponent = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))
        coef = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_cov))
        pdf_value = coef * np.exp(exponent)

        return round(pdf_value.item(), 19)
