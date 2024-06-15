#!/usr/bin/env python3
"""
    This module implements Normalization
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes.
    """
    return (X - m) / s
