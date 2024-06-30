#!/usr/bin/env python3
'''
    This module returns the location of a user
    with a github api-url.
'''
import requests
import sys
from datetime import datetime, timedelta

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <API_URL>")
        sys.exit(1)

    api_url = sys.argv[1]

    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            user_data = response.json()
            print(user_data.get('location', 'Location not specified'))
        elif response.status_code == 403:
            reset_time = datetime.fromtimestamp(int(response.headers.get('X-RateLimit-Reset', 0)))
            now = datetime.now()
            minutes_remaining = (reset_time - now).total_seconds() // 60
            print('Reset in {} min'.format(int(minutes_remaining)))
        elif response.status_code == 404:
            print('Not found')
        else:
            print('Error: {}'.format(response.status_code))
    except requests.exceptions.RequestException as e:
        print('Request failed: {}'.format(e))
