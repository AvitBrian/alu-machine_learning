#!/usr/bin/env python3
"""
Test file
"""
user_location = __import__('2-user_location').get_user_location
location = user_location('https://api.github.com/users/holbertonschool/')

print(location)