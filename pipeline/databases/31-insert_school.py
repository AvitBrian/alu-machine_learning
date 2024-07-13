#!/usr/bin/env python3
"""
    This function that inserts a new document in
    a MongoDB collection
"""


def insert_school(mongo_collection, **kwargs):
    """
    Inserts a new document in a MongoDB collection.

    """
    document = mongo_collection.insert_one(kwargs)
    return (document.inserted_id)
