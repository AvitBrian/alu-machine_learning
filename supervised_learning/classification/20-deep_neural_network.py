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
        """
            Initializes the deep neural network object.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        self.nx = nx
        self.layers = layers

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.weights["W1"] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx))
            else:
                self.weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]
                ) * np.sqrt(2 / layers[i - 1])
            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

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

    def forward_prop(self, X):
        '''Calculates the forward propagation'''
        self.__cache["A0"] = X
        for i in range(self.__L):
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            A = self.__cache["A{}".format(i)]
            Z = np.matmul(W, A) + b
            self.__cache["A{}".format(i + 1)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        '''Calculates the cost of the model'''
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the deep neural network'''
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, cost
