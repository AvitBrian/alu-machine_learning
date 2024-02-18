#!/usr/bin/env python3
"""
this function performs convolution on grayscale images with various options.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs convolution on grayscale images.

    Args:
        images (numpy.ndarray): Input grayscale images with shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).
        padding (tuple or str): Padding option for the convolution.
        stride (tuple): Stride for the convolution operation.

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(((height - 1) * stride[0] + kh - height) / 2) + 1
        pw = int(((width - 1) * stride[1] + kw - width) / 2) + 1
    else:
        ph, pw = padding

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    h_res = (h - kh + 2 * ph) // sh + 1
    w_res = (w - kw + 2 * pw) // sw + 1
    convolved_images = np.zeros((m, h_res, w_res))

    for i in range(h_res):
        for j in range(w_res):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i*sh:i*sh + kh,
                              j*sw:j*sw + kw] * kernel, axis=(1, 2))

    return convolved_images
