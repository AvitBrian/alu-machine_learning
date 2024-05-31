#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the neural network weights.
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        dW = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if layer > 1:
            D_prev = cache['D' + str(layer - 1)]
            A_prev = cache['A' + str(layer - 1)]
            dz = np.matmul(W.T, dz) * (1 - A_prev ** 2)
            dz *= D_prev
            dz /= keep_prob

        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db
