#!/usr/bin/env python3
'''
Bidirectional Cell Forward
'''


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    '''
    Function that performs forward propagation for a bidirectional RNN
    '''
    t, m, i = X.shape
    h = h_0.shape[1]
    
    Hf = np.zeros((t, m, h))  
    Hb = np.zeros((t, m, h))
    Hf[0] = h_0 
    Hb[-1] = h_t  

    # Forward pass
    for step in range(1, t):
        Hf[step] = bi_cell.forward(Hf[step - 1], X[step])

    # Backward pass
    for step in range(t - 2, -1, -1):
        Hb[step] = bi_cell.backward(Hb[step + 1], X[step])

    H = np.concatenate((Hf, Hb), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
