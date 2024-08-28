# import tensorflow as tf
# from .upsample import Upsample
# from src import logger

# class Generator(tf.keras.Model):
#     def __init__(self): 
#         super(Generator, self).__init__()
#         self.model = self.build()
#         if self.model is None:
#             logger.error("Generator model is None after build")
#         else:
#             logger.info("Generator model built successfully")

#     def build(self):
#         logger.info("Building Generator Model")
#         generator_input = tf.keras.Input(shape=(16, 16, 512))
#         x = Upsample(256, 4, apply_batch_normalization=True)(generator_input)
#         x = Upsample(128, 4)(x)
#         x = Upsample(64, 4)(x)
#         x = Upsample(32, 4)(x)
#         generator_output = tf.keras.layers.Conv2D(
#             filters=3,
#             kernel_size=4,
#             padding='same',
#             activation='tanh',
#             kernel_initializer='he_normal'
#         )(x)
#         model = tf.keras.Model(generator_input, generator_output, name="generator")
#         logger.info("Generator model built")
#         logger.info(f"Generator model summary: {model.summary()}")
#         return model

#     def call(self, inputs):
#         if self.model is None:
#             logger.error("Generator model is None in call method")
#             raise ValueError("Generator model is not initialized")
#         return self.model(inputs)


# import tensorflow as tf
# from .upsample import Upsample
# from src import logger

# class Generator(tf.keras.Model):
#     def __init__(self): 
#         super(Generator, self).__init__()
#         self.upsample1 = Upsample(256, 4, apply_batch_normalization=True)
#         self.upsample2 = Upsample(128, 4)
#         self.upsample3 = Upsample(64, 4)
#         self.upsample4 = Upsample(32, 4)
#         self.output_layer = tf.keras.layers.Conv2D(
#             filters=3,
#             kernel_size=4,
#             padding='same',
#             activation='tanh',
#             kernel_initializer='he_normal'
#         )
#         logger.info("Generator layers initialized")

#     def call(self, inputs):
#         x = self.upsample1(inputs)
#         x = self.upsample2(x)
#         x = self.upsample3(x)
#         x = self.upsample4(x)
#         return self.output_layer(x)

#     def build_graph(self, input_shape=(16, 16, 512)):
#         x = tf.keras.Input(shape=input_shape)
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))


import tensorflow as tf
from .upsample import Upsample
from src import logger

class Generator(tf.keras.Model):
    def __init__(self): 
        super(Generator, self).__init__()
        self.upsample1 = Upsample(256, 4, apply_batch_normalization=True)
        self.upsample2 = Upsample(128, 4)
        self.upsample3 = Upsample(64, 4)
        self.upsample4 = Upsample(32, 4)
        self.output_layer = tf.keras.layers.Conv2D(
            filters=3,  # Ensure this is 3 for RGB output
            kernel_size=4,
            padding='same',
            activation='tanh',
            kernel_initializer='he_normal'
        )
        logger.info("Generator layers initialized")

    def call(self, inputs):
        x = self.upsample1(inputs)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        return self.output_layer(x)

    def build_graph(self, input_shape=(16, 16, 512)):
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
