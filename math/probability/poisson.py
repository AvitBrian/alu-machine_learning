#!/usr/bin/env python3
'''
    Poisson distribution
    that represents a poisson distribution
'''


class Poisson:
    '''
    Class Poisson that represents a distribution of Poisson.

    Attributes:
        lambtha (float): The average number of events in a given time period.

    Methods:
        factorial(k): Calculates the factorial of a given number.
        __init__(data, lambtha): Class constructor.
        pmf(k): Calculates the value of the Probability Mass Function (PMF) for a given number of successes.
        cdf(k): Calculates the value of the Cumulative Distribution Function (CDF) for a given number of successes.
    '''

    def factorial(self, k):
        '''
        Calculates the factorial of a given number.

        Args:
            k (int): The number for which the factorial is calculated.

        Returns:
            int: The factorial of the given number.
        '''
        if k < 0:
            return 0
        if k == 0 or k == 1:
            return 1
        return k * self.factorial(k - 1)

    def __init__(self, data=None, lambtha=1.):
        '''
        Class constructor.

        Args:
            data (list, optional): List of data points. If provided, the average (lambtha) will be calculated based on the data points. Defaults to None.
            lambtha (float, optional): The average number of events in a given time period. Defaults to 1.

        Raises:
            ValueError: If lambtha is not a positive value.
            ValueError: If data does not contain multiple values.
            TypeError: If data is not a list.
        '''
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        '''
        Calculates the value of the Probability Mass Function (PMF) for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The value of the PMF for the given number of successes.
        '''
        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285
        pmf = self.lambtha**k * e**(-self.lambtha)/self.factorial(k)
        return pmf

    def cdf(self, k):
        '''
        Calculates the value of the Cumulative Distribution Function (CDF) for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The value of the CDF for the given number of successes.
        '''
        cdf = 0
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
