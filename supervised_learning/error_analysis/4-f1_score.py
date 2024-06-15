#!/usr/bin/env python3
"""
    This module calculates the F1 score of a confusion matrix.
"""
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    calculates the F1 score in a confusion matrix.
    """
    return 2 * precision(confusion) * sensitivity(confusion) / \
        (precision(confusion) + sensitivity(confusion))
