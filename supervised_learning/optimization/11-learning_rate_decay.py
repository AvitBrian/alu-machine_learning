#!/usr/bin/env python3
"""
    This module deals with Learnin decay.
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the ;earning rate decay using inverse decay.
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
