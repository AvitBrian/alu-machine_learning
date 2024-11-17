#! /usr/bin/env python3
'''
This script contains the implementation of the policy gradient algorithm.
'''
import numpy as np

def policy(matrix, weight):
    '''
    Computes the policy with a weight matrix
    Args:
        matrix: state matrix of shape (s, f) where s is number of states and f is number of features
        weight: weight matrix of shape (f, a) where f is number of features and a is number of actions
    Returns:
        policy matrix of shape (s, a) containing probabilities for each action in each state
    '''
    # Compute z = state * weight matrix multiplication
    z = np.matmul(matrix, weight)
    
    # np.exp applies exponential function e^x to each element in z
    # This is part of the softmax calculation to convert raw scores to probabilities
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on state and weight matrix
    Args:
        state: matrix representing current observation of environment
        weight: matrix of random weight
    Returns:
        action: selected action
        gradient: computed policy gradient
    """
    action_probs = policy(state, weight)    
    action = np.random.choice(len(action_probs[0]), p=action_probs[0])

    one_hot = np.zeros_like(action_probs)
    one_hot[0][action] = 1

    gradient = np.matmul(state.T, (one_hot - action_probs))

    return action, gradient
