#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""
import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout."""
    dense_layer = tf.layers.dense(prev, units=n, activation=activation)
    dropout = tf.layers.Dropout(rate=1 - keep_prob)
    return dropout(dense_layer)
