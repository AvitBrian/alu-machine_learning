#!/usr/bin/env python3
"""
Creates a TF-IDF embedding
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding
    """
    tfidf = TfidfVectorizer(vocabulary=vocab)
    X = tfidf.fit_transform(sentences)
    features = tfidf.get_feature_names()
    embeddings = X.toarray()

    return embeddings, features
