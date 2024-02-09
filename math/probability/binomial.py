#!/usr/bin/env python3


class Binomial:
    """
    A class representing the Binomial distribution.

    Attributes:
        n (int): The number of trials.
        p (float): The probability of success.

    Methods:
        __init__(self, data=None, n=1, p=0.5):
        Initializes a Binomial instance.
        calculate_parameters(self, data):
        Calculates the parameters of the Binomial distribution.
        pmf(self, k): Calculates the probability mass function (PMF)
        for a given value of k.
        cdf(self, k): Calculates the cumulative distribution function (CDF)
        for a given value of k.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes a Binomial object.

        Parameters:
        - data (list): A list of values.
        - n (int): The number of trials (default is 1).
        - p (float): The probability of success (default is 0.5).

        Raises:
        - ValueError: If n is less than 1 or if p is not between 0 and 1.
        - TypeError: If data is not a list.
        - ValueError: If data contains less than 2 values.
        """
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.n, self.p = self.calculate_parameters(data)

    def calculate_parameters(self, data):
        """
        Calculates the parameters for abinomial distribution
        based on the given data.

        Parameters:
        - data (list): A list of numerical values representing the data.

        Returns:
        - tuple: A tuple containing the calculated parameters (n, p), where:
            - n (int): The number of trials.
            - p (float): The probability of success in each trial.
        """
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        q = variance / mean
        p = 1 - q
        n = round(mean / p)
        p = mean / n
        return n, p

    def pmf(self, k):
        """
        Calculates the probability mass function (PMF)
        of the binomial distribution for a given value of k.

        Parameters:
        - k (int): The number of successes in the binomial distribution.

        Returns:
        - pmf (float): The probability mass function value for
        the given value of k.
        """
        k = int(k)
        if k < 0:
            return 0
        p = self.p
        q = 1 - p
        binomial_co = 1
        for i in range(1, k + 1):
            binomial_co *= (self.n - i + 1) / i
        pmf = binomial_co * (p ** k) * (q ** (self.n - k))
        return pmf

    def cdf(self, k):
        """
        Calculates the cumulative distribution function (CDF)
        of the binomial distribution.

        Parameters:
        - k (int): The number of successes.

        Returns:
        - cdf (float): The cumulative probability
        of getting k or fewer successes.
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = sum(self.pmf(i) for i in range(k + 1))
        return cdf
