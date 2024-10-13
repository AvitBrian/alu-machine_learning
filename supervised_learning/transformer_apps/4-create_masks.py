'''
This module contains the create_masks function for creating attention masks
in a transformer model for machine translation.
'''

import tensorflow as tf


def create_masks(inputs, target):
    '''
    Creates all masks for training/validation in a transformer model.
    '''
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder target padding mask
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((tf.shape(target)[1], tf.shape(target)[1])), -1, 0)
    dec_target_padding_mask = tf.cast(
        tf.math.equal(target, 0), tf.float32)
    dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis,
                                                      tf.newaxis, :]
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
