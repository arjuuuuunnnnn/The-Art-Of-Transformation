from tensorflow import keras
from src import logger
from tensorflow.keras import layers
import tensorflow as tf

class Downsample(keras.layers.Layer):
    def __init__(self, filters, size, apply_batch_normalization=True):
        super(Downsample, self).__init__()
        self.filters = filters
        self.size = size
        self.apply_batch_normalization = apply_batch_normalization
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=size,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal'
        )
        if apply_batch_normalization:
            self.batch_norm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        logger.info(f"Downsample layer created with {filters} filters and size {size}")

    def call(self, inputs):
        x = self.conv(inputs)
        if self.apply_batch_normalization:
            x = self.batch_norm(x)
        return self.leaky_relu(x)
