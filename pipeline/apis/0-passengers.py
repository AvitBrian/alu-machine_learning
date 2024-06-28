#!/usr/bin/env python3
"""
    This module returns he list of ships
    that can hold a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """
    Returns the list of ships.
    """
    url = "https://swapi-api.alx-tools.com/api/starships/"
    ships = []
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for each_ship in data["results"]:
            passengers = each_ship['passengers'].replace(',', '')
            if (passengers == 'n/a' or 
                int(passengers) >= passengerCount):
                ships.append(each_ship['name'])

    url = data["next"]
    return ships
