#!/usr/bin/env python3
"""
This module the list of names
of the home planets of all sentient species.
"""
import requests


def sentientPlanets():
    """
    Returns a list of planet names.
    """
    url = "https://swapi-api.alx-tools.com/api/species/"
    sentient_planets = []
    while url:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            for species in data['results']:
                classification = species['classification']
                designation = species['designation']
                homeworld = species['homeworld']
                if (classification == 'sentient' or
                        designation == 'sentient') and homeworld:
                    planet = requests.get(homeworld).json()
                    sentient_planets.append(planet['name'])
            url = data["next"]
    return sentient_planets
