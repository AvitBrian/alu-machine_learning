#!/usr/bin/env python3
'''Initialize Q-table for OpenAI Gym environment'''


import numpy as np
import gym


def q_init(env):
    '''
    Initialize Q-table with zeros based on 
    environment state/action space
    '''
    return np.zeros((env.observation_space.n, env.action_space.n))
