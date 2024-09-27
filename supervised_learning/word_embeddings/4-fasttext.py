#!/usr/bin/env python3
"""
Creates and trains a gensim FastText model
"""
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a gensim FastText model
    """
    if cbow is True:
        skip = 0
    else:
        skip = 1
    model = FastText(size=size, window=window,
                     min_count=min_count, workers=workers, sg=skip,
                     negative=negative, seed=seed)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=iterations)
    return model
