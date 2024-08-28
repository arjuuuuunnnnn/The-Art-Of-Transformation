# import tensorflow as tf
# from tensorflow import keras
# from .downsample import Downsample
# from src import logger

# class Discriminator(tf.keras.Model):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = self.build()

#     def build(self):
#         discriminator_input = tf.keras.Input(shape=(256, 256, 3))
#         x = Downsample(64, 4, apply_batch_normalization=False)(discriminator_input)
#         x = Downsample(128, 4, apply_batch_normalization=True)(x)
#         x = Downsample(256, 4, apply_batch_normalization=False)(x)
#         x = Downsample(512, 4)(x)
#         discriminator_output = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(x)
#         model = tf.keras.Model(discriminator_input, discriminator_output, name="discriminator")
#         logger.info("Discriminator model built")
#         return model

#     def __call__(self, inputs):
#         return self.model(inputs)


# import tensorflow as tf
# from .downsample import Downsample
# from src import logger

# class Discriminator(tf.keras.Model):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         # Define layers in __init__
#         self.downsample1 = Downsample(64, 4, apply_batch_normalization=False)
#         self.downsample2 = Downsample(128, 4, apply_batch_normalization=True)
#         self.downsample3 = Downsample(256, 4, apply_batch_normalization=False)
#         self.downsample4 = Downsample(512, 4)
#         self.conv = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')

#     def call(self, inputs):
#         x = self.downsample1(inputs)
#         x = self.downsample2(x)
#         x = self.downsample3(x)
#         x = self.downsample4(x)
#         discriminator_output = self.conv(x)
#         return discriminator_output


import tensorflow as tf
from .downsample import Downsample
from src import logger

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.downsample1 = Downsample(64, 4, apply_batch_normalization=False)
        self.downsample2 = Downsample(128, 4, apply_batch_normalization=True)
        self.downsample3 = Downsample(256, 4, apply_batch_normalization=False)
        self.downsample4 = Downsample(512, 4)
        self.output_layer = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')
        logger.info("Discriminator layers initialized")

    def call(self, inputs):
        x = self.downsample1(inputs)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.downsample4(x)
        return self.output_layer(x)

    def build_graph(self, input_shape=(256, 256, 3)):
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
