#!/usr/bin/env python3
"""
    This class represents a single neuron performing
    binary classification.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
        class: DeepNeuralNetwork
    """

    """
        class: DeepNeuralNetwork
    """
    def __init__(self, nx, layers, activation='sig'):

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:

                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        ''' return the L attribute'''
        return self.__L

    @property
    def cache(self):
        ''' return the cache attribute'''
        return self.__cache

    @property
    def weights(self):
        ''' return the weights attribute'''
        return self.__weights

    def forward_prop(self, X):
        """ Forward propagation"""
        self.cache["A0"] = X
        for i in range(1, self.L+1):
            # extract values
            W = self.weights['W'+str(i)]
            b = self.weights['b'+str(i)]
            A = self.cache['A'+str(i - 1)]
            # do forward propagation
            z = np.matmul(W, A) + b
            if i != self.L:
                A = 1 / (1 + np.exp(-z))  # sigmoid function
            else:
                A = np.exp(z) / np.sum(np.exp(z), axis=0)  # softmax function
            # store output to the cache
            self.cache["A"+str(i)] = A
        return self.cache["A"+str(i)], self.cache

    def cost(self, Y, A):
        """ Calculate the cost of the Neural Network \
        """
        cost = -np.sum(Y * np.log(A)) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """ Evaluate the neural network
        """
        self.forward_prop(X)
        # get output of the neural network from the cache
        A = self.cache.get("A" + str(self.L))
        # get the class with the highest probability
        prediction = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculate one pass of gradient descent on the neural network
        """
        m = Y.shape[1]

        for i in range(self.L, 0, -1):

            A_prev = cache["A" + str(i - 1)]
            A = cache["A" + str(i)]
            W = self.__weights["W" + str(i)]

            if i == self.__L:
                dz = A - Y
            else:
                dz = da * (A * (1 - A))
            db = dz.mean(axis=1, keepdims=True)
            dw = np.matmul(dz, A_prev.T) / m
            da = np.matmul(W.T, dz)
            self.__weights['W' + str(i)] -= (alpha * dw)
            self.__weights['b' + str(i)] -= (alpha * db)

    def train(self, X, Y, iterations=5000,
            alpha=0.05, verbose=True, graph=True, step=100):
        """ Train the deep neural network
        """

        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        costs = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if verbose and i % step == 0:

                cost = self.cost(Y, self.cache["A"+str(self.L)])
                costs.append(cost)
                print('Cost after {} iterations: {}'.format(i, cost))
        if graph:
            plt.plot(np.arange(0, iterations, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Save the instance object to a file in pickle format
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """ Load a pickled DeepNeuralNetwork object
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None