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
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
