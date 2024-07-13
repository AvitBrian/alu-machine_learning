#!/usr/bin/env python3
"""
    This function returns the list of schools with a specific topic.
"""


def schools_by_topic(mongo_collection, topic):
    """
    Finds list of all schools with a specific topic.
    """
    documents = mongo_collection.find({'topics': {'$all': [topic]}})
    schools = []
    for every_doc in documents:
        schools.append(every_doc)
    return schools
