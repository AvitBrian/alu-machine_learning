#!/usr/bin/env python3
'''
This module implements PCA to reduce
'''
import numpy as np


def pca(X, var=0.95):
    """
    This function implements PCA to reduce
    the dimensionality of the input data,
    while maintaining a specified fraction of the original variance.
    """
    # covariance matrix,eigenvalues and eigenvectors
    cov_matrix = np.cov(X.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance

    n_components = np.argmax(cumulative_variance_ratio >= var) + 1

    W = eigenvectors[:, :n_components]

    return W
