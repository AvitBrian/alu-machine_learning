#!/usr/bin/env python3
"""
This module performs Neural Style Transfer on two images
"""
import numpy as np
import tensorflow as tf

class NST:
    """Neural Style Transfer class"""
    
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'
    
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        
        tf.enable_eager_execution()
        
        self.style_image = style_image
        self.content_image = content_image
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1 
        and its largest side is 512 pixels
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        
        h, w, _ = image.shape
        max_dim = max(h, w)
        scale = 512 / max_dim
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Resize the image
        image = tf.image.resize(image[tf.newaxis, :], (new_h, new_w), 
                                method='bicubic', preserve_aspect_ratio=True)
        
        # Ensure the image is clipped between 0 and 1
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
