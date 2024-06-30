#!/usr/bin/env python3
"""
This module retrieves information about the upcoming SpaceX launch.
"""
import requests
from datetime import datetime, timedelta, timezone

def get_launch_info(launch_data):
    """
    Retrieves and formats the launch information.
    """
    launch_name = launch_data["name"]
    date_unix = launch_data["date_unix"]
    rocket_id = launch_data["rocket"]
    launchpad_id = launch_data["launchpad"]

    launch_date_utc = datetime.fromtimestamp(date_unix, tz=timezone.utc)
    launch_date_local = launch_date_utc.astimezone(timezone(timedelta(hours=-4)))
    launch_date_str = launch_date_local.strftime('%Y-%m-%dT%H:%M:%S%z')
    launch_date_str = "{}:{}".format(launch_date_str[:-2], launch_date_str[-2:])

    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_response = requests.get(rocket_url)
    rocket_name = rocket_response.json()["name"]

    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_response = requests.get(launchpad_url)
    launchpad_data = launchpad_response.json()
    launchpad_name = launchpad_data["name"]
    launchpad_locality = launchpad_data["locality"]

    return "{} ({}) {} - {} ({})".format(
        launch_name,
        launch_date_str,
        rocket_name,
        launchpad_name,
        launchpad_locality
    )

def get_upcoming_launch():
    """
    Retrieves information about the upcoming SpaceX launch.
    """
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)
    launches = response.json()
    launches.sort(key=lambda x: x["date_unix"])
    return get_launch_info(launches[0])

if __name__ == "__main__":
    print(get_upcoming_launch())
