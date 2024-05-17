#!/usr/bin/env python3
"""
This class represents a single neuron performing binary classification.
"""
import numpy as np


class Neuron:
    """
    Class: Neuron
    """
    def __init__(self, nx):
        """
        Constructor for Neuron class.

        Args:
            nx (int): Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is not a positive integer.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for the weights.

        Returns:
            numpy.ndarray: The weights of the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the bias.

        Returns:
            float: The bias of the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the activated output.

        Returns:
            float: The activated output of the neuron.
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            float: The activated output of the neuron.
        """
        self.__A = 1 / (1 + np.exp(-np.dot(self.__W, X) - self.__b))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): True labels.
            A (numpy.ndarray): Predicted labels.

        Returns:
            float: The cost.
        """
        m = Y.shape[1]
        cost = ((-1 / m) * np.sum(Y * np.log(A) + (1 - Y)
                                  * np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.

        Returns:
            numpy.ndarray: The neuron's predictions.
            float: The cost.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.
            A (numpy.ndarray): Predicted labels.
            alpha (float): Learning rate.

        Returns:
            numpy.ndarray: The updated weights.
            float: The updated bias.
        """
        m = Y.shape[1]
        dz = A - Y
        db = (1 / m) * np.sum(dz)
        dw = (1 / m) * np.matmul(X, dz.T)
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)
        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.
            iterations (int, optional): Number of iterations to train over.
            alpha (float, optional): Learning rate.

        Raises:
            TypeError: If iterations is not an integer.
            ValueError: If iterations is not a positive integer.
            TypeError: If alpha is not a float.
            ValueError: If alpha is not positive.

        Returns:
            numpy.ndarray: The neuron's predictions after training.
            float: The cost after training.
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
