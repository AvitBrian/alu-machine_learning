#!/usr/bin/env python3
"""
    This module trains a neural network.
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    function: create_placeholders
    creates placeholders x and y
    @nx: the number of feature columns in our data
    @classes: the number of classes in our classifier
    Return: placeholders named x and y, respectively
    """
    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='y')
    return x, y
