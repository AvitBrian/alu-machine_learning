#!/usr/bin/env python3
"""
Creates a word count or binary word presence embedding
"""
from sklearn.feature_extraction.text import CountVectorizer

def tf_idf(sentences, vocab=None):
    """
    Creates a word count or binary presence embedding
    """
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, binary=True)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names_out()
    embeddings = X.toarray()

    return embeddings, features
