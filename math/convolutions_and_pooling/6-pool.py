#!/usr/bin/env python3
"""
this function performs pooling on images.
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images (numpy.ndarray): Input images with shape (m, h, w, c).
        kernel_shape (tuple): Kernel shape for the pooling (kh, kw).
        stride (tuple): Stride for the pooling operation (sh, sw).
        mode (str): Type of pooling ('max' or 'avg').

    Returns:
        numpy.ndarray: Pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1
    pooled_images = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                pooled_images[:, i, j, :] = np.max(
                    images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, i, j, :] = np.mean(
                    images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2))

    return pooled_images
