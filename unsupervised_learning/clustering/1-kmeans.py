#!/usr/bin/env python3
'''
This module implements K-means clustering algorithm
'''
import numpy as np


def kmeans_plus_plus_init(X, k):
    '''
    Initializes cluster centroids for K-means using K-means++ method
    '''
    n, d = X.shape
    centroids = np.zeros((k, d))
    centroids[0] = X[np.random.randint(n)]

    for i in range(1, k):
        dist_sq = np.min(np.sum((X[:, np.newaxis] - centroids[:i])**2, axis=2),
                         axis=1)
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        centroids[i] = X[np.searchsorted(cumulative_probs, r)]

    return centroids


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

    C = kmeans_plus_plus_init(X, k)

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
