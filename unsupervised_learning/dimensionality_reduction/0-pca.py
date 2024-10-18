#!/usr/bin/env python3
'''
This module implements PCA to reduce dimensionality
'''
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    cumulative_variance_ratio = np.cumsum(S**2) / np.sum(S**2)

    n_components = np.argmax(cumulative_variance_ratio >= var) + 1

    W = Vt.T[:, :n_components]

    return W
