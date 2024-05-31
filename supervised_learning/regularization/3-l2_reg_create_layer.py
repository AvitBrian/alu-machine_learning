#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Creates a tensorflow layer that includes L2 regularization. """
    return tf.layers.dense(prev, units=n, activation=activation,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(lambtha))