import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from src.Transform.utils import ImageUtils
from src import logger

class Tester:
    def __init__(self):
        self.encoder, self.generator = self.load_models()

    def load_models(self):
        encoder = keras.models.load_model('encoder_model')
        generator = keras.models.load_model('generator_model')
        logger.info("Models loaded for testing")
        return encoder, generator

    def generate_sketch(self, image_path):
        image = ImageUtils.preprocess_image(image_path)
        latent_code = self.encoder(image, training=False)
        generated_sketch = self.generator(lat
