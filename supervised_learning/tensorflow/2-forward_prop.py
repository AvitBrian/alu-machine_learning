#!/usr/bin/env python3
"""
    This module performs forward propagation,
    without importing any modules.
"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes, activations):
    '''
        calculates the forward propagation.
    '''
    op = x
    for size, activation in zip(layer_sizes, activations):
        op = create_layer(op, size, activation)
    return op