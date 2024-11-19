#!/usr/bin/env python3
'''Load and return the FrozenLakeEnv from OpenAI gym'''

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    '''Load and return the FrozenLakeEnv environment'''
    return gym.make('FrozenLake-v0',
                    desc=desc,
                    map_name=map_name,
                    is_slippery=is_slippery)
