#!/usr/bin/env python3
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
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        if response.status_code == 404:
            print("Not found")
        elif response.status_code == 403:
            reset_time = datetime.fromtimestamp(int(response.headers['X-RateLimit-Reset']))
            now = datetime.now()
            minutes_remaining = (reset_time - now).total_seconds() // 60
            print(f"Reset in {int(minutes_remaining)} min")
        else:
            print(e)
        sys.exit(1)

    user_data = response.json()
    location = user_data.get('location', 'Not available')
    print(location)