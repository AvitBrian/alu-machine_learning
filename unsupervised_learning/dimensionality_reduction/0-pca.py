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

    for i in range(eigenvectors.shape[1]):
        max_abs_idx = np.argmax(np.abs(eigenvectors[:, i]))
        if eigenvectors[max_abs_idx, i] < 0:
            eigenvectors[:, i] *= -1

    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance

    n_components = np.argmax(cumulative_variance_ratio >= var) + 1

    W = eigenvectors[:, :n_components]

    return W
