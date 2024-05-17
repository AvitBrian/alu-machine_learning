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
        '''Calculates the forward propagation'''
        self.__A1 = np.matmul(self.__W1, X)+self.__b1
        self.__A1 = 1 / (1 + np.exp(-self.__A1))
        self.__A2 = np.matmul(self.__W2, self.__A1)+self.__b2
        self.__A2 = 1 / (1 + np.exp(-self.__A2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''Calculates the cost of the model'''
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) *
                        np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neural network's predictions'''
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''Calculates one pass of gradient descent'''
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (1 / m) * np.matmul(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * np.matmul(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)

        return self.__W1, self.__b1, self.__W2, self.__b2
