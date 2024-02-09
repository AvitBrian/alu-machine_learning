class Binomial:
    """
    A class representing the Binomial distribution.

    Attributes:
        trials (int): The number of trials.
        success_prob (float): The probability of success.

    Methods:
        __init__(self, data=None, trials=1, success_prob=0.5): Initializes a Binomial instance.
        calculate_parameters(self, data): Calculates the parameters of the Binomial distribution.
        pmf(self, k): Calculates the probability mass function (PMF) for a given value of k.
        cdf(self, k): Calculates the cumulative distribution function (CDF) for a given value of k.
    """

    def __init__(self, data=None, trials=1, success_prob=0.5):
        if data is None:
            if trials < 1:
                raise ValueError("trials must be a positive value")
            if not 0 < success_prob < 1:
                raise ValueError("success_prob must be greater than 0 and less than 1")
            self.trials = trials
            self.success_prob = success_prob
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.trials, self.success_prob = self.calculate_parameters(data)

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
        p = self.success_prob
        q = 1 - p
        binomial_co = 1
        for i in range(1, k + 1):
            binomial_co *= (self.trials - i + 1) / i
        pmf = binomial_co * (p ** k) * (q ** (self.trials - k))
        return pmf

    def cdf(self, k):
        k = int(k)
        if k < 0:
            return 0
        cdf = sum(self.pmf(i) for i in range(k + 1))
        return cdf
