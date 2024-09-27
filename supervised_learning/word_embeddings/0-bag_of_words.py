#!/usr/bin/env python3
"""
this module contains the function bag_of_words
"""

import numpy as np


def bag_of_words(sentences, vocab=None):
    '''
    creates a bag of words embedding matrix
    '''
    if vocab is None:
        vocab = set()
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            vocab.add(word)
    vocab = sorted(list(vocab))
    embeddings = np.zeros((len(sentences), len(vocab)))
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for j, word in enumerate(vocab):
            embeddings[i, j] = words.count(word)

    return embeddings, vocab