#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early
    """
    if cost >= opt_cost - threshold:
        count += 1
        if count == patience:
            return (True, count)
        return (False, count)
    if cost < opt_cost - threshold:
        opt_cost = cost
        count = 0
        return (False, count)
