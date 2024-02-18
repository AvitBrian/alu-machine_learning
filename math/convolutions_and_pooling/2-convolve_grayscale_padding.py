#!/usr/bin/env python3
"""
this function performs convolution on grayscale images with custom padding.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): Input grayscale images with shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).
        padding (tuple): Padding for height and width of the image (ph, pw).

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           'constant',0)

    h_res = h + 2 * ph - kh + 1
    w_res = w + 2 * pw - kw + 1
    convolved_images = np.zeros((m, h_res, w_res))

    for i in range(h_res):
        for j in range(w_res):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))

    return convolved_images
