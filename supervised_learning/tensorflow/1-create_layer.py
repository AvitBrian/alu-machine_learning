#!/usr/bin/env python3
"""
    This module creates a neural network layer.
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    function: create_layer
    creates a neural network layer
    @prev: is the tensor output of the previous layer
    @n: is the number of nodes in the layer to create
    @activation: is the activation function that the layer should use
    Return: the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation, kernel_initializer=init,
                            name='layer')
    return layer(prev)
