#!/usr/bin/env python3
'''
    This class represents a single neuron performing
    binary classification.
'''
import numpy as np


class Neuron:
    '''
        Class: Neuron
    '''
    def __init__(self, nx):
        '''
            Constructor for Neuron class.
            Args:
                nx (int): Number of input features.

            Raises:
                TypeError: If nx is not an integer.
                ValueError: If nx is not a positive integer.
        '''
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''
            Getter for the weights.
            Returns:
                numpy.ndarray: The weights of the neuron.
        '''
        return self.__W

    @property
    def b(self):
        '''
            Getter for the bias.
            Returns:
                float: The bias of the neuron.
        '''
        return self.__b

    @property
    def A(self):
        '''
            Getter for the activated output.
            Returns:
                float: The activated output of the neuron.
        '''
        return self.__A

    def forward_prop(self, X):
        '''
            Calculates the forward propagation of the neuron.
            Args:
                X (numpy.ndarray): Input data.
            Returns:
                float: The activated output of the neuron.
        '''
        self.__A = 1 / (1 + np.exp(-np.dot(self.__W, X) - self.__b))
        return self.__A
