#!/usr/bin/env python3
"""
    This module performs forward propagation.
"""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    function: forward_prop
    performs forward propagation
    @x: is the input data
    @layer_sizes: is a list containing the number of nodes in each layer
    of the network
    @activations: is a list containing the activation functions for each
    layer of the network
    Return: the prediction of the network in tensor form
    """
    for i in range(len(layer_sizes)):
        init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        layer = tf.layers.Dense(layer_sizes[i], activations[i],
                                kernel_initializer=init, name='layer')
        x = layer(x)
    return x