#!/usr/bin/env python3
'''
    a class Exponential that represents
    an exponential distribution
'''


class Exponential:
    '''
    Class Exponential that represents
    an exponential distribution

    Attributes:
    - data: A list of values representing the data points
    - lambtha: A float representing the rate parameter
    of the exponential distribution

    Methods:
    - __init__(self, data=None, lambtha=1.): 
    Initializes the Exponential object
    - pdf(self, x): Calculates the value of the PDF
    for a given time period
    - cdf(self, x): Calculates the value of the CDF
    for a given time period
    '''
    def __init__(self, data=None, lambtha=1.):
        '''
        Class constructor

        Parameters:
        - data: A list of values representing
        the data points (default: None)
        - lambtha: A float representing the rate parameter
        of the exponential distribution (default: 1.)

        Raises:
        - ValueError: If lambtha is not a positive value
        - ValueError: If data does not contain multiple values
        - TypeError: If data is not a list
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
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        '''
        Calculates the value of the PDF for a given time period

        Parameters:
        - x: A float representing the time period

        Returns:
        - The value of the PDF at the given time period
        '''
        if x < 0:
            return 0
        
        pdf = self.lambtha * (2.7182818285 ** (-self.lambtha * x))
        return pdf

    def cdf(self, x):
        '''
        Calculates the value of the CDF for a given time period

        Parameters:
        - x: A float representing the time period

        Returns:
        - The value of the CDF at the given time period
        '''
        if x < 0:
            return 0
        cdf = 1 - (2.7182818285 ** (-self.lambtha * x))
        return cdf
