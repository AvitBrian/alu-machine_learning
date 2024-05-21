#!/usr/bin/env python3
"""
    This module performs forward propagation,
    without importing any modules.
"""
create_layer = __import__('1-create_layer').create_layer
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x is the placeholder for the input data
    layer_sizes is a list containing the number of nodes in each layer of the network
    activations is a list containing the activation functions for each layer of the network
    Returns: the prediction of the network in tensor form
    """
    prediction = x
    for i in range(len(layer_sizes)):
        prediction = create_layer(prediction,
                                  layer_sizes[i], activations[i])
    return prediction
