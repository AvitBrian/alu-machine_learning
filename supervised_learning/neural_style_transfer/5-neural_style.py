#!/usr/bin/env python3
"""
This module performs Neural Style Transfer on two images
"""
import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize variables and model"""
        tf.enable_eager_execution()

        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape "
                            "(h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape "
                            "(h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales image to range [0, 1] with the largest side of 512 px"""
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape "
                            "(h, w, 3)")

        scale = 512 / max(image.shape[:-1])
        new_shape = tuple(map(lambda x: int(scale * x), image.shape[:-1]))
        image = image[tf.newaxis, :]
        image = tf.image.resize_bicubic(image, new_shape)
        image = image / 255
        return tf.clip_by_value(image, 0, 1)

    def load_model(self):
        """Loads the VGG19 model for cost calculation"""
        base_vgg = tf.keras.applications.VGG19(include_top=False,
                                               weights='imagenet')
        custom_object = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        base_vgg.save('base_vgg')
        vgg = tf.keras.models.load_model('base_vgg',
                                         custom_objects=custom_object)
        for layer in vgg.layers:
            layer.trainable = False
        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates Gram matrix for input layer"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        num_locations = tf.cast(input_layer.shape[1] * input_layer.shape[2],
                                tf.float32)
        return result / num_locations

    def generate_features(self):
        """Extracts style and content features"""
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        style_outputs = self.model(style_image)[:-1]
        content_output = self.model(content_image)[-1]
        self.gram_style_features = [self.gram_matrix(style_output)
                                    for style_output in style_outputs]
        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """Calculates style cost for a single layer"""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or len(
                style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        if gram_target.shape != (1, style_output.shape[-1],
                                 style_output.shape[-1]):
            raise TypeError(f"gram_target must be a tensor of shape [1, "
                            f"{style_output.shape[-1]}, "
                            f"{style_output.shape[-1]}]")
        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculates the total style cost"""
        if not isinstance(style_outputs, list) or len(style_outputs) != len(
                self.style_layers):
            raise TypeError(f"style_outputs must be a list with a length of "
                            f"{len(self.style_layers)}")

        weight = 1 / len(self.style_layers)
        style_costs = [self.layer_style_cost(output, target) * weight
                       for output, target in zip(style_outputs,
                                                 self.gram_style_features)]
        return tf.add_n(style_costs)