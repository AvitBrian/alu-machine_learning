#!/usr/bin/env python3
'''
this module deals Montecarlo implementation
'''

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    '''Implements Monte Carlo on a given environment'''
    for _ in range(episodes):
        state = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state, action) not in [(s, a) for s, a, _ in episode[0:t]]:
                V[state] = V[state] + alpha * (G - V[state])
    return V
