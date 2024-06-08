#!/usr/bin/env python3
"""
    This module implements Normalization
"""
import numpy as np


def normalization_constants(X):
    """ Calculates the normalization constants. """
    return X.mean(axis=0), X.std(axis=0)
