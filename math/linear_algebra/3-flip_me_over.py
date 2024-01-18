#!/usr/bin/env python3
def matrix_transpose(matrix):
    T = []
    for row in zip(*matrix):
        T.append(list(row))
    return T
