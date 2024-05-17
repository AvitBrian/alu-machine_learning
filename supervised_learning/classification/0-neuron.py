#!/usr/bin/env python3
'''
    This class represents a single neuron performing
    binary classification.
'''
import numpy as np


class Neuron:
    '''
    Class that represents a single neuron performing binary classification.

    Attributes:
        W (numpy.ndarray): The weights vector for the neuron.
        It has shape (1, nx).
        b (int): The bias for the neuron.
        A (int): The activated output of the neuron (prediction).
    '''
    def __init__(self, nx):
        '''
        Initializes a Neuron instance.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is not a positive integer.
        '''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
