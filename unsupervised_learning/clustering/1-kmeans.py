#!/usr/bin/env python3
'''
This module implements K-means clustering algorithm
'''
import numpy as np


def initialize(X, k):
    '''
    Initializes cluster centroids for K-means
    '''

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None

    n, d = X.shape
    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)
    centroid = np.random.uniform(minimum, maximum, size=(k, d))

    return centroid


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

    C = initialize(X, k)

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        new_C = np.array([X[clss == i].mean(axis=0) if np.sum(clss == i) > 0
                          else X[np.random.choice(len(X))]
                          for i in range(k)])
        if np.allclose(C, new_C):
            break
        C = new_C

    return C, clss
