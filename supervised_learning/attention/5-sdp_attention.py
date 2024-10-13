#!/usr/bin/env python3
'''
This module contains the sdp_attention function.
'''
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention
    args:
        Q [tensor of shape (..., seq_len_q, dk)]:
            query matrix
        K [tensor of shape (..., seq_len_k, dk)]:
            key matrix
        V [tensor of shape (..., seq_len_v, dv)]:
            value matrix
        mask:
            mask to be applied for multi-head attention
    Returns:
        output [tensor of shape (..., seq_len_q, dv)]:
            output of the scaled dot product attention
        weights [tensor of shape (..., seq_len_q, seq_len_k)]:
            weights of the scaled dot product attention
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_qk = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_qk += (mask * -1e9)
    weights = tf.nn.softmax(scaled_qk, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
