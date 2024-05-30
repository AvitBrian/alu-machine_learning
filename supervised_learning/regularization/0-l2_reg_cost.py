#!/usr/bin/env python3
"""
    This module performs l2 regularization.
"""


def l2_reg_cost(cost, lambtha, weights, L, m):
    l2_cost = 0
    for i in range(1, L + 1):
        weight = weights['W' + str(i)]
        l2_cost += np.linalg.norm(weight)**2
    l2_cost *= lambtha / (2 * m)
    return cost + l2_cost
