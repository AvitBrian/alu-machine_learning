#!/usr/bin/env python3
class Binomial:
    """
    A class representing the Binomial distribution.

    Attributes:
        n (int): The number of trials.
        p (float): The probability of success.

    Methods:
        __init__(self, data=None, n=1, p=0.5): Initializes a Binomial instance.
        calculate_parameters(self, data): Calculates the parameters of the Binomial distribution.
        pmf(self, k): Calculates the probability mass function (PMF) for a given value of k.
        cdf(self, k): Calculates the cumulative distribution function (CDF) for a given value of k.
    """

    def __init__(self, data=None, n=1, p=0.5):
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
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        q = variance / mean
        p = 1 - q
        n = round(mean / p)
        p = mean / n
        return n, p

    def pmf(self, k):
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
        k = int(k)
        if k < 0:
            return 0
        cdf = sum(self.pmf(i) for i in range(k + 1))
        return cdf
