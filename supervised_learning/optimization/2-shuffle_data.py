#!/usr/bin/env python3
"""
    This module implements Normalization
"""
import numpy as np


def shuffle_data(X, Y):
    """shuffling with permutations"""
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]
