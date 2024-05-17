#!/usr/bin/env python3
"""
    This class represents a single neuron performing
    binary classification.
"""
import numpy as np


def one_hot_decode(one_hot):
    """
        one hot decode function
        Args: one_hot
        Returns: numpy.ndarray
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    vector = one_hot.transpose().argmax(axis=1)
    return vector
