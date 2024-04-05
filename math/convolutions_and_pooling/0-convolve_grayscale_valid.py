import numpy as np

#!/usr/bin/env python3
"""
This function performs convolutions on grayscale images using the "valid" padding strategy.
"""



def convolve_grayscale_valid(images, kernel):
    """
    Performs convolutions on grayscale images using the "valid" padding strategy.

    Args:
        images (numpy.ndarray): Input grayscale images with shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).

    Returns:
        numpy.ndarray: Convolved images with shape (m, output_h, output_w), where
                       output_h = h - kh + 1 and output_w = w - kw + 1.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1  # Valid padding: output height = input height - kernel height + 1
    output_w = w - kw + 1  # Valid padding: output width = input width - kernel width + 1

    convolved_images = np.zeros((m, output_h, output_w))  # Pre-allocate output with correct shape

    for i in range(output_h):
        for j in range(output_w):
            # Element-wise multiplication and summation within the valid region
            convolved_images[:, i, j] = np.sum(images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))

    return convolved_images

