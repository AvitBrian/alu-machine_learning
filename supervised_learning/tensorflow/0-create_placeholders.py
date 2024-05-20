#!/usr/bin/env python3
"""
    This module returns place holders x and y.
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns: placeholders named x and y, respectively
    """
    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='y')
    return x, y
