#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Creates a tensorflow layer that includes L2 regularization. """
    ker = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=ker,
                            kernel_regularizer=reg)
    return layer(prev)
