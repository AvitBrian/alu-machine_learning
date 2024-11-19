#!/usr/bin/env python3
'''Determine next action using epsilon-greedy algorithm'''


import numpy as np 


def epsilon_greedy(Q, state, epsilon):
    '''Determine next action using epsilon-greedy algorithm'''
    p = np.random.uniform(0, 1)
    # exploring and exploiting
    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action