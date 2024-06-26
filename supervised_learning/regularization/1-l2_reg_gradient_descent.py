#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Calculates gradient descent. """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i-1)]
        dw = (1 / m) * np.matmul(dz, A_prev.T) + \
             (lambtha / m) * weights['W' + str(i)]
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        if i > 1:
            da = 1 - cache['A' + str(i-1)]**2
            dz = np.matmul(weights['W' + str(i)].T, dz) * da
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
