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

    def __init__(self, nx, layers):
        ''' DeepNeuralNetwork class constructor'''
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.nx = nx
        self.layers = layers

        # Initialize weights and biases and validate layers in one loop
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights["W1"] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx))
            else:
                self.__weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]
                ) * np.sqrt(2 / layers[i - 1])
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    # create the getter functions of the deep network
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
        ''' Forward propagation of the neural network '''
        self.__cache["A0"] = X
        for i in range(self.__L):
            Z = np.matmul(self.__weights["W{}".format(i + 1)], self.__cache["A{}".format(i)]) + self.__weights["b{}".format(i + 1)]
            if i == self.__L - 1:
                self.__cache["A{}".format(i + 1)] = self.softmax(Z)
            else:
                self.__cache["A{}".format(i + 1)] = self.relu(Z)
        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        ''' Cost calculation for softmax '''
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        ''' Evaluation of the neural network '''
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.argmax(A, axis=0)
        return prediction, cost

    def softmax(self, Z):
        '''
            Softmax activation function
        '''
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def relu(self, x):
        '''Relu activation function'''
        return np.maximum(0, x)

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''Gradient descent method'''
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y
        for i in reversed(range(self.__L)):
            A_prev = cache["A{}".format(i)]
            W = self.__weights["W{}".format(i + 1)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 0:
                A = cache["A{}".format(i)]
                dZ = np.matmul(W.T, dZ) * (1 - A)

    def train(
        self, X, Y, iterations=5000,
        alpha=0.05, verbose=True, graph=True, step=100
    ):
        '''
            Trains the deep neural network
        '''
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if graph or verbose:
            if type(step) is not int:
                raise TypeError('step must be an integer')

            if step < 1 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        cost_list = []

        for i in range(iterations):
            # Forward propagation
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            cost_list.append(cost)

            # Gradient descent
            self.gradient_descent(Y, cache, alpha)
            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(np.arange(0, iterations + 1, step), cost_list)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        '''
            Saves the instance object to a file
        '''
        if type(filename) is not str:
            return
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        '''
            Loads the file with the model
        '''
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                return obj
        except FileNotFoundError:
            return None
