#!/usr/bin/env python3
"""
    This module deals with batch normalization.
"""


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a
    neural Network.
    """
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    Z_norm = (Z - mean) / ((variance + epsilon) ** 0.5)
    Z_out = gamma * Z_norm + beta
    return Z_out
