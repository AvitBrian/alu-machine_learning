#!/usr/bin/env python3
"""
    This module performs forward propagation,
    without importing any modules.
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x is the placeholder for the input data
    layer_sizes is a list containing the number of nodes in each layer of the network
    activations is a list containing the activation functions for each layer of the network
    Returns: the prediction of the network in tensor form
    """
    for i in range(len(layer_sizes)):
        if i == 0:
            layer = x
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
