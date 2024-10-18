#!/usr/bin/env python3
'''
This module implements PCA to reduce
the dimensionality of input data
'''
import numpy as np


def pca(X, ndim):
    """
    This function implements PCA to reduce
    the dimensionality of the input data
    to a specified number of dimensions.
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    cov_matrix = np.cov(X_centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    W = eigenvectors[:, :ndim]

    T = np.matmul(X_centered, W)

    return T
