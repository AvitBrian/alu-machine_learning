#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    layer = tf.layers.Dense(
        units=n,  kernel_initializer=initializer, activation=activation
    )
    result = layer(prev)
    dense_layer = tf.layers.dropout(result, rate=1 - keep_prob)

    return dense_layer
