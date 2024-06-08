#!/usr/bin/env python3
"""
    This module deals with RMSprop Optimization.
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates variables with the RMSProp optimization algorithm.
    """
    s = beta2 * s + (1 - beta2) * np.power(grad, 2)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
