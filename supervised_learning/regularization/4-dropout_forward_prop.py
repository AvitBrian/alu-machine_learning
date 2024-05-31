#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs the dropout forward propagation.
    """
    cache = {}
    cache["A0"] = X
    for i in range(L):
        A = cache["A" + str(i)]
        W = weights["W" + str(i + 1)]
        b = weights["b" + str(i + 1)]

        Z = np.matmul(W, A) + b

        if i == L - 1:
            t = np.exp(Z)
            cache["A" + str(i + 1)] = t / np.sum(
                t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = ((np.random.rand(A.shape[0], A.shape[1])
                  < keep_prob).astype(int))
            A *= D
            A /= keep_prob
            cache["A" + str(i + 1)] = A
            cache["D" + str(i + 1)] = D

    return cache
