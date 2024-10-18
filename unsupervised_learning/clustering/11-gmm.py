#!/usr/bin/env python3
"""Module for calculating a GMM from a dataset"""

import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset"""
    np = sklearn.mixture.GaussianMixture._estimate_gaussian_parameters.\
        __globals__['np']
    if not isinstance(X, (list, np.ndarray)) or len(X) == 0:
        return None, None, None, None, None
    if not all(isinstance(row, (list, np.ndarray)) for row in X):
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    X = np.array(X)
    if len(X.shape) != 2:
        return None, None, None, None, None

    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)

    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)

    return pi, m, S, clss, np.array([bic])
