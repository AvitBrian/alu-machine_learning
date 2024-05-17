#!/usr/bin/env python3
'''
    This class represents a single neuron performing
    binary classification.
'''
import numpy as np


class NeuralNetwork:
    '''
        class: NeuralNetwork
    '''

    def __init__(self, nx, nodes):
        '''
            Initializes the neural network object.

            Args:
                nx (int): Number of input features.
                nodes (int): Number of nodes in the hidden layer.

            Raises:
                TypeError: If nx or nodes is not an integer.
                ValueError: If nx or nodes is less than 1.

        '''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')

        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''Getter method for the weight matrix W1.'''
        return self.__W1

    @property
    def b1(self):
        '''Getter method for the bias vector b1.'''
        return self.__b1

    @property
    def A1(self):
        '''Getter method for the activated output A1.'''
        return self.__A1

    @property
    def W2(self):
        '''Getter method for the weight matrix W2.'''
        return self.__W2

    @property
    def b2(self):
        '''Getter method for the bias scalar b2.'''
        return self.__b2

    @property
    def A2(self):
        '''Getter method for the activated output A2.'''
        return self.__A2

    def forward_prop(self, X):
        '''
            Calculates the forward propagation of the neural network
        '''
        self.__A1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-self.__A1))
        self.__A2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-self.__A2))
        return self.__A1, self.__A2
