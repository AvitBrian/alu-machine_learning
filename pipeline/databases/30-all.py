#!/usr/bin/env python3
"""
This Function lists all documents in MongoDB collection
"""


def list_all(mongo_collection):
    """
    Lists all documents in a MongoDB collection
    """
    all_docs = []
    collection = mongo_collection.find()
    for document in collection:
        all_docs.append(document)
    return all_docs
