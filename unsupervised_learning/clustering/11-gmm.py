#!/usr/bin/env python3
"""Module for calculating a GMM from a dataset"""

import sklearn.mixture
import numpy as np


def gmm(X, k):
    """Calculates a GMM from a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)

    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)

    return pi, m, S, clss, np.array([bic])
