#!/usr/bin/env python3
'''
This script contains a function that performs
semantic search on a corpus of documents.
'''
import os
from sentence_transformers import SentenceTransformer, util

def semantic_search(corpus_path, sentence):
    """Perform semantic search on a corpus of documents."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    max_similarity = -1
    most_similar_text = ""

    for filename in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, filename), 'r') as file:
            document = file.read()
            doc_embedding = model.encode(document, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(sentence_embedding, doc_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_text = document

    return most_similar_text
