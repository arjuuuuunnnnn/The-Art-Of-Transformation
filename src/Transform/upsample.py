from tensorflow import keras
from src import logger
import tensorflow as tf

class Upsample(keras.layers.Layer):
    def __init__(self, filters, size, apply_dropout=True, apply_batch_normalization=True):
        super(Upsample, self).__init__()
        self.filters = filters
        self.size = size
        self.apply_dropout = apply_dropout
        self.apply_batch_normalization = apply_batch_normalization
        
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=size,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal'
        )
        if apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.1)
        if apply_batch_normalization:
            self.batch_norm = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        logger.info(f"Upsample layer created with {filters} filters and size {size}")

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        if self.apply_dropout:
            x = self.dropout(x)
        if self.apply_batch_normalization:
            x = self.batch_norm(x)
        return self.leaky_relu(x)
