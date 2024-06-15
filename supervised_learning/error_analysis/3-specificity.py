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
        false_positives = np.sum(confusion[:, i]) - true_positives
        false_negatives = np.sum(confusion[i, :]) - true_positives

        sensitivity = true_positives / (true_positives + false_negatives)
        precision[i] = true_positives / (true_positives + false_positives)
        specificity[i] = precision[i] * sensitivity \
            / (precision[i] + sensitivity)

    return sensitivity, precision
