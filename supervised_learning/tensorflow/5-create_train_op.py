#!/usr/bin/env python3
"""
    This module creates a training operation.
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    function: create_train_op
    creates a training operation
    @loss: is the loss of the network's prediction
    @alpha: is the learning rate
    Return: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
