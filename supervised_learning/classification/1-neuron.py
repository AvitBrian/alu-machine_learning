#!/usr/bin/env python3
'''
    This class represents a single neuron performing
    binary classification.
'''
import numpy as np


class Neuron:
    '''
    Class Neuron represents a single neuron for binary classification.
    '''
    def __init__(self, nx):
        '''
        Initializes a Neuron object.

        Args:
            nx (int): The number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is not a positive integer.
        '''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''
        Getter method for the weights.

        Returns:
            numpy.ndarray: The weights of the neuron.
        '''
        return self.__W

    @property
    def b(self):
        '''
        Getter method for the bias.

        Returns:
            float: The bias of the neuron.
        '''
        return self.__b

    @property
    def A(self):
        '''
        Getter method for the activation output.

        Returns:
            float: The activation output of the neuron.
        '''
        return self.__A
