#!/usr/bin/env python3
"""
This class represents a single neuron performing
binary classification.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
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

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A = self.__cache['A' + str(i - 1)]
            Z = np.matmul(W, A) + b
            if i != self.__L:
                A = 1 / (1 + np.exp(-Z))
            else:
                A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
            self.__cache["A" + str(i)] = A
        return A, self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.argmax(A, axis=0)
        Y_argmax = np.argmax(Y, axis=0)
        accuracy = np.sum(predictions == Y_argmax) / Y.shape[1]
        return accuracy, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        A = cache["A{}".format(self.__L)]
        dZ = A - Y
        for i in reversed(range(self.__L)):
            A = cache["A{}".format(i + 1)]
            A_prev = cache["A{}".format(i)]
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.matmul(W.T, dZ) * A_prev * (1 - A_prev)
            self.__weights["W{}".format(i + 1)] -= alpha * dW
            self.__weights["b{}".format(i + 1)] -= alpha * db
            self.__cache["A{}".format(i)] = A
        return self.__weights, self.__cache

    def train(
        self, X, Y, iterations=5000,
        alpha=0.05, verbose=True, graph=True, step=100
    ):
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
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            cost_list.append(cost)
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
        if type(filename) is not str:
            return
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                return obj
        except FileNotFoundError:
            return None
