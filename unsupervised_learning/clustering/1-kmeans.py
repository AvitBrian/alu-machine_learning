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
    C = np.zeros((k, d))
    
    C[0] = X[np.random.randint(n)]

    for i in range(1, iterations + k):
        if i < k:
            # K-means++ initialization
            D = np.min(np.sum((X[:, np.newaxis] - C[:i])**2, axis=2), axis=1)
            prob = D / D.sum()
            cumprob = np.cumsum(prob)
            r = np.random.rand()
            C[i] = X[np.searchsorted(cumprob, r)]
        else:
            # K-means algorithm
            distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
            clss = np.argmin(distances, axis=1)
            new_C = np.array([X[clss == j].mean(axis=0) if np.sum(clss == j) > 0
                              else X[np.random.choice(len(X))]
                              for j in range(k)])
            if np.allclose(C, new_C):
                break
            C = new_C

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
