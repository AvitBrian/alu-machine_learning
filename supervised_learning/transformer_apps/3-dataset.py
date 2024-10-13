"""
This module contains the Dataset class, which handles dataset loading,
tokenization, encoding, and pipeline setup for Portuguese to English
translation.
"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Handles dataset loading, tokenization, encoding, and pipeline setup
    """

    def __init__(self, batch_size, max_len):
        """
        Initialize the dataset with given batch size and max length.
        """
        dataset, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)
        self.data_train, self.data_valid = dataset['train'], dataset['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        # Update data_train attribute
        self.data_train = self.data_train.filter(
            lambda x, y: tf.logical_and(
                tf.size(x) <= max_len,
                tf.size(y) <= max_len
            )
        )
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(
            self.data_train.cardinality().numpy())
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        # Update data_valid attribute
        self.data_valid = self.data_valid.filter(
            lambda x, y: tf.logical_and(
                tf.size(x) <= max_len,
                tf.size(y) <= max_len
            )
        )
        self.data_valid = self.data_valid.padded_batch(batch_size)

    def tokenize_dataset(self, data):
        """
        Build tokenizers for Portuguese and English from the training data.
        """
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encode a translation pair into tokens.
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode method.
        """
        result_pt, result_en = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
