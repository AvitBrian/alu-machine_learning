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

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(min_vals, max_vals, size=(k, d))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        
        for i in range(k):
            cluster_points = X[clss == i]
            if len(cluster_points) > 0:
                C[i] = np.mean(cluster_points, axis=0)
            else:
                C[i] = np.random.uniform(min_vals, max_vals)

        if np.allclose(C, C): 
            break

    return C, clss
