#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """ Calculates reg losses. """
    reg_losses = tf.losses.get_regularization_losses()
    return cost + reg_losses
