#!/usr/bin/env python3
"""
    This module deals with RMSprop Optimization.
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates RMSProp optimization algorithm.
    """
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
