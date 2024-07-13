#!/usr/bin/env python3
"""
This script Displays Nginx logs stored in MongoDB
"""
from pymongo import MongoClient


def log_stats():
    """
    Returns stats about Nginx logs stored in MongoDB
    """
    client = MongoClient()
    db = client['logs']
    collection = db['nginx']

    total_logs = collection.count_documents({})
    print(f"{total_logs} logs")

    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    count = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{count} status check")

if __name__ == "__main__":
    log_stats()
