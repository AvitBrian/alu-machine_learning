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
            for each_species in data["results"]:
                species_class = each_species["classification"].lower()
                if all([species_class == "sentient",
                    each_species["homeworld"] != None]):
                    homeworld_url = each_species["homeworld"]
                    homeworld_response = requests.get(homeworld_url)
                    homeworld_data = homeworld_response.json()
                    sentient_planets.append(homeworld_data["name"])

        url = data["next"]

    return sentient_planets
