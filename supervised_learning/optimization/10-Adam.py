#!/usr/bin/env python3
"""
    This module deals with Adam Optimization.
"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates Adam optimization."""
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return optimizer.minimize(loss)
