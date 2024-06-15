#!/usr/bin/env python3
"""
    This module calculates specificity
    for each class in a confusion matrix.
"""
import numpy as np


def specificity(confusion):
    """
    calculates the specificity in a confusion matrix.
    """
    classes = confusion.shape[0]
    specificity = np.zeros(classes)
    precision = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i, i]
        true_negatives = np.sum(confusion) - np.sum(confusion[i, :]) \
            - np.sum(confusion[:, i]) + true_positives
        false_positives = np.sum(confusion[:, i]) - true_positives
        false_negatives = np.sum(confusion[i, :]) - true_positives

        specificity[i] = true_negatives / (true_negatives + false_positives)

    return specificity
