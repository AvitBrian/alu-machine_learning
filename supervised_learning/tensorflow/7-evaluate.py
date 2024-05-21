#!/usr/bin/env python3
"""
    This module evaluates the output of a neural network.
"""
import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op

def evaluate(X, Y, save_path):
    """
    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from
    Returns: the network’s prediction, accuracy, and loss
    """
    x, y = create_placeholders(X.shape[1], Y.shape[1])
    y_pred = forward_prop(x, [256, 256, 10], ['relu', 'relu', 'softmax'])
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})
    return prediction, accuracy, loss

