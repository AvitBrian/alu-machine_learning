#!/usr/bin/env python3
"""
This modules Performs Neural Style Transfer on two images
"""
import numpy as np
import tensorflow as tf

class NST:
  style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1','block4_conv1', 'block5_conv1']
  content_layer = 'block5_conv2'
  def __init__(self, style_image, content_image, alpha=1e4, beta=1):
    if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
      raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
    if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
      raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
    if alpha < 0:
      raise TypeError("alpha must be a non-negative number")
    if beta < 0:
      raise TypeError("beta must be a non-negative number")
    
    tf.enable_eager_execution()
    
    self.style_image = style_image
    self.content_image = style_image
    self.alpha = alpha
    self.beta = beta

def scale_image(image):
  """
  Scales an Image
  
  image -- The image to be scaled
  Returns -- scaled image.
  """
  if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
    raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
  image = tf.convert_to_tensor(image, dtype=tf.float32)
  image = tf.convert_to_tensor(image, dtype=tf.float32)
  h,w,_ = image.shape
  if h > w:
    h_new = 512
    w_new = int(w * h_new / h)
  else:
    w_new = 512
    h_new = int(h * w_new / w)
    
  image = tf.image.resize_bicubic(image, (h_new, w_new), preserve_aspect_ratio=True)
  image = tf.expand_dims(image, axis=0)
  return image
