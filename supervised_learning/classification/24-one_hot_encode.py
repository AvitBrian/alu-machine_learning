#!/usr/bin/env python3
"""
    This class represents a single neuron performing
    binary classification.
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
        function: one hot encode
        Args: Y, classes
        Returns:
            one-hot encoding of Y with shape (classes, m)
    """
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        one_hot = np.eye(classes)[Y].transpose()
        return one_hot
    except Exception as err:
        return None
