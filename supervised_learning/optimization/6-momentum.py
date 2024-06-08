#!/usr/bin/env python3
"""
    This modules deals with momentum optimization.
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates Momentum optimization.
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
