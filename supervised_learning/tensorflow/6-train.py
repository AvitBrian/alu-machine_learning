#!/usr/bin/env python3
"""
    This module trains the neural network.
"""
import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    function: train
    trains a neural network
    """
    tf.reset_default_graph()

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('placeholders', x)
    tf.add_to_collection('placeholders', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('tensors', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('tensors', loss)

    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('operation', train_op)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('tensors', accuracy)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            _, train_cost = sess.run([train_op, loss], feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

        save_path = saver.save(sess, save_path)
        print("Model saved in path: %s" % save_path)

    return save_path
