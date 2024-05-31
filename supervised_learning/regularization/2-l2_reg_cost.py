#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""
import tensorflow as tf


def l2_reg_cost(cost):
    reg_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
    return cost + sum(reg_losses)
