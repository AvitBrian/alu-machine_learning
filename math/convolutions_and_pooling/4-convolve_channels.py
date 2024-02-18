#!/usr/bin/env python3
"""
this function does convolution on images with channels and various options.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs convolution on images with channels.

    Args:
        images (numpy.ndarray): Input images with shape (m, h, w, c).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw, c).
        padding (tuple or str): Padding option for the convolution.
        stride (tuple): Stride for the convolution operation.

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    else:
        ph, pw = padding

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           'constaSSnt', 0)

    h_res = (h + 2 * ph - kh) // sh + 1
    w_res = (w + 2 * pw - kw) // sw + 1
    convolved_images = np.zeros((m, h_res, w_res))

    for i in range(0, h_res):
        for j in range(0, w_res):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i*sh:i*sh + kh,
                              j*sw:j*sw + kw] * kernel, axis=(1, 2, 3))

    return convolved_images
