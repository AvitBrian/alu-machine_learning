#!/usr/bin/env python3
"""
    This class represents a single neuron performing
    binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
        class: DeepNeuralNetwork
    """
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if i == 0:
                self.__weights['W' + str(i+1)] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2/nx)
                )
            else:
                self.__weights['W' + str(i+1)] = (
                    np.random.randn(layers[i], layers[i-1]) * np.sqrt(
                        2/layers[i-1]
                    )
                )
            self.__weights['b' + str(i+1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter method for the L attribute."""
        return self.__L

    @property
    def cache(self):
        """Getter method for the cache attribute."""
        return self.__cache

    @property
    def weights(self):
        """Getter method for the weights attribute."""
        return self.__weights
