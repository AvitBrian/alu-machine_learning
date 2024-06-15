import numpy as np
import tensorflow as tf

def shuffle_data(X, Y):
    """
    Shuffles the data in a matrix.
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    X_shuffled = X[shuffle]
    Y_shuffled = Y[shuffle]
    return X_shuffled, Y_shuffled

def calculate_loss(y, y_pred):
    """
    Calculates the cross-entropy loss of a prediction.
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss

def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction in a DNN.
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def create_layer(prev, n, activation):
    """
    Creates a TF layer.
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=initializer, name="layer")
    return layer(prev)

def create_batch_norm_layer(prev, n, activation):
    """
    Normalizes a batch in a DNN with Tf.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n, kernel_initializer=init)
    x_prev = x(prev)
    scale = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma")
    mean, variance = tf.nn.moments(x_prev, axes=[0])
    offset = tf.Variable(tf.constant(0.0, shape=[n]), name="beta")
    variance_epsilon = 1e-8

    normalization = tf.nn.batch_normalization(
        x_prev,
        mean,
        variance,
        offset,
        scale,
        variance_epsilon,
    )
    if activation is None:
        return normalization
    return activation(normalization)

def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Performs forward propagation in a DNN.
    """
    layer = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        if i != len(layer_sizes) - 1:
            layer = create_batch_norm_layer(layer, layer_sizes[i], activations[i])
        else:
            layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer

def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Trains a DNN with TF RMSProp optimization.
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
    return optimizer

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Performs learning rate decay in TensorFlow using inverse time decay.
    """
    LRD = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, staircase=True)
    return LRD

def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path="/tmp/model.ckpt"):
    """
    Trains a DNN model.
    """