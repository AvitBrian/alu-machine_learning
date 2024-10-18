#!/usr/bin/env python3
'''
This module implements K-means clustering algorithm
'''
import numpy as np


def kmeans(X, k, iterations=1000):
    '''
    Performs K-means clustering on a dataset
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize cluster centroids
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(min_vals, max_vals, size=(k, d))

    for _ in range(iterations):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        new_C = np.array([X[clss == i].mean(axis=0) for i in range(k)])

        # Check for empty clusters and reinitialize if necessary
        empty_clusters = np.where(np.isnan(new_C).any(axis=1))[0]
        if empty_clusters.size > 0:
            new_C[empty_clusters] = np.random.uniform(
                min_vals, max_vals, size=(len(empty_clusters), d))

        # Check for convergence
        if np.allclose(C, new_C):
            return C, clss

        C = new_C

    return C, clss
