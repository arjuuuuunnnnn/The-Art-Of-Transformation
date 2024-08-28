import tensorflow as tf
from .downsample import Downsample
from src import logger

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_input = tf.keras.Input(shape=(256, 256, 3), name="img")
        self.d1 = Downsample(32, 4, apply_batch_normalization=False)
        self.d2 = Downsample(64, 4, apply_batch_normalization=False)
        self.d3 = Downsample(128, 4, apply_batch_normalization=False)
        self.d4 = Downsample(256, 4)
        self.conv = tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu')

        self.build((None, 256, 256, 3))
        logger.info("Encoder model built")

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        encoder_output = self.conv(x)
        return encoder_output

