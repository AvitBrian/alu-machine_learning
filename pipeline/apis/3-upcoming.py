#!/usr/bin/env python3
"""
This module retrieves information about the upcoming SpaceX launch.
"""
import requests
from datetime import datetime, timezone


def get_upcoming_launch():
    """
    Retrieves information about the upcoming SpaceX launch.
    """
    try:
        launches_response = requests.get('https://api.spacexdata.com/v4/launches/upcoming/')
        if launches_response.status_code == 200:
            upcoming_launches = launches_response.json()
            next_launch = sorted(upcoming_launches, key=lambda x: x['date_unix'])[0]

            launch_name = next_launch['name']
            launch_date_utc = datetime.fromtimestamp(next_launch['date_unix'], timezone.utc)
            launch_date_local = launch_date_utc.astimezone().strftime('%Y-%m-%d %H:%M:%S')
            rocket_id = next_launch['rocket']

            rocket_response = requests.get(f'https://api.spacexdata.com/v4/rockets/{rocket_id}/')
            rocket_name = rocket_response.json().get('name', 'Unknown rocket')

            launchpad_id = next_launch['launchpad']
            launchpad_response = requests.get(f'https://api.spacexdata.com/v4/launchpads/{launchpad_id}/')
            launchpad_data = launchpad_response.json()
            launchpad_name = launchpad_data.get('name', 'Unknown launchpad')
            launchpad_location = launchpad_data.get('locality', 'Unknown locality')

            print(f'{launch_name} ({launch_date_local}) {rocket_name} - {launchpad_name} ({launchpad_location})')
        else:
            print(f'Error: {species_response.status_code}')
    except requests.exceptions.RequestException as error:
        print(f'Request failed: {error}')

get_upcoming_launch()
