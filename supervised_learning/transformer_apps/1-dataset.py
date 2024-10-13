'''
This module contains the Dataset class, which handles dataset loading,
tokenization, and encoding for Portuguese to English translation.
'''
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    """
    Handles dataset loading, tokenization,
    and encoding for Portuguese to English translation.
    """

    def __init__(self):
        dataset, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)
        self.data_train = dataset['train']
        self.data_valid = dataset['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        '''
        Build tokenizers for Portuguese and English from the training data.
        '''
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        '''
        Encode Portuguese and English sentences using the respective tokenizers.
        '''
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        return np.array(pt_tokens), np.array(en_tokens)
