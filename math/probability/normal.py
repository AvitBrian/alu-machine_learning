#!/usr/bin/env python3
'''
    this class is for Normal distribution
'''


class Normal:
    '''
    Class Normal that represents
    a normal distribution
    '''

    def __init__(self, data=None, mean=0., stddev=1.):
        '''
        Class constructor

        Parameters:
        - data (list): List of data points to calculate
        mean and standard deviation from
        - mean (float): Mean value of the
        normal distribution (default: 0.0)
        - stddev (float): Standard deviation value of
        the normal distribution (default: 1.0)

        Raises:
        - ValueError: If stddev is a negative value
        - TypeError: If data is not a list
        - ValueError: If data contains less than 2 values
        '''
        if data is None:
            if stddev < 1:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = ((sum((x - self.mean) ** 2 for x in data)/len(data))
                           ** 0.5)

    def z_score(self, x):
        '''
        Calculates the z-score of a given x-value

        Parameters:
        - x (float): The value to calculate the z-score for

        Returns:
        - float: The z-score of the given x-value
        '''
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        '''
        Calculates the x-value of a given z-score

        Parameters:
        - z (float): The z-score to calculate the x-value for

        Returns:
        - float: The x-value of the given z-score
        '''
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        '''
        Calculates the value of the PDF for a given x-value

        Parameters:
        - x (float): The value to calculate the PDF for

        Returns:
        - float: The value of the PDF for the given x-value
        '''
        coefficient = 1 / (self.stddev * (2 * 3.1415926536) ** 0.5)
        power = -0.5 * ((x - self.mean) / self.stddev) ** 2
        pdf = coefficient * 2.7182818285 ** power
        return round(pdf, 10)

    def cdf(self, x):
        '''
        Calculates the value of the CDF for a given x-value

        Parameters:
        - x (float): The value to calculate the CDF for

        Returns:
        - float: The value of the CDF for the given x-value
        '''
        value = self.z_score(x) / (2 ** 0.5)
        val = value - ((value ** 3) / 3) + ((value ** 5) / 10)
        val = val - ((value ** 7) / 42) + ((value ** 9) / 216)
        cdf = 0.5 * (1 + val) * (2 / (3.1415926536 ** 0.5))
        return cdf
