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
        url = 'https://api.spacexdata.com/v4/launches/upcoming'
        response = requests.get(url)
        if response.status_code == 200:
            launches = response.json()
            upcoming_launch = sorted(launches, key=lambda x: x['date_unix'])[0]

            launch_name = upcoming_launch['name']
            launch_date_local = upcoming_launch['date_local']
            rocket_id = upcoming_launch['rocket']

            sub_url = 'https://api.spacexdata.com/v4/rockets/'
            rocket_response = requests.get(sub_url + str(rocket_id))
            rocket_name = rocket_response.json().get('name', 'Unknown rocket')

            launchpad_id = upcoming_launch['launchpad']
            sub_pad = 'https://api.spacexdata.com/v4/launchpads/'
            launchpad_response = requests.get(sub_pad + str(launchpad_id))
            launchpad_data = launchpad_response.json()
            launchpad_name = launchpad_data.get('name', 'Unknown launchpad')
            launch_local = launchpad_data.get('locality', 'Unknown locality')

            output = "{} ({}) {} - {} ({})".format(
                launch_name,
                launch_date_local,
                rocket_name,
                launchpad_name,
                launch_local
            )
            print(output)
        else:
            print('Error: ' + str(response.status_code))
    except requests.exceptions.RequestException as e:
        print('Request failed: ' + str(e))


if __name__ == '__main__':
    get_upcoming_launch()
