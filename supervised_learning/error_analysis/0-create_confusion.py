#!/usr/bin/env python3
"""
    This modules creates a confusion matrix.
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix.
    """
    return np.matmul(labels.T, logits)
