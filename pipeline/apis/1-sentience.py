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
    planets = []
    classisfication = "sentient"
    while url is not None:
        response = requests.get(url)
        data = response.json()
        for species in data["results"]:
            if all([
                species["classification"].lower() == classisfication,
                species["homeworld"]]
            ):
                homeworld = species["homeworld"]
                res_homeworld = requests.get(homeworld)
                data_homeword = res_homeworld.json()
                planets.append(data_homeword["name"])

        url = data["next"]
    return planets

