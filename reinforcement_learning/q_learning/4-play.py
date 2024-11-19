#!/usr/bin/env python3
'''Play an episode using a trained agent'''


import gym
import numpy as np


def play(env, Q, max_steps=100):
    '''Have a trained agent play one episode'''

    current_state = env.reset()
    done = False
    env.render()

    for each_step in range(max_steps):
        action = np.argmax(Q[current_state, :])
        next_state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
        current_state = next_state

    return reward
