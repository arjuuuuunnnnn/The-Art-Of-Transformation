import tensorflow as tf
import matplotlib.pyplot as plt
from src import logger

class ImageUtils:
    @staticmethod
    def normalize(arr):
        arr_min = tf.reduce_min(arr)
        arr_max = tf.reduce_max(arr)
        arr_range = arr_max - arr_min
        scaled = (arr - arr_min) / arr_range
        return -1 + (scaled * 2)

    @staticmethod
    def denormalize(image):
        return (image + 1.0) * 0.5

    @staticmethod
    def show_images(images, sketches, num_images=2):
        plt.figure(figsize=(10, 5))
        for i in range(min(num_images, len(images))):
            plt.subplot(2, num_images, i + 1)
            plt.imshow(ImageUtils.denormalize(images[i]))
            plt.title("Image")
            plt.axis("off")
            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(ImageUtils.denormalize(sketches[i]), cmap="gray")
            plt.title("Sketch")
            plt.axis("off")
        plt.show()
        logger.info(f"Displayed {num_images} image-sketch pairs")

    @staticmethod
    def preprocess_image(image_path):
        try:
            # Read the file
            image = tf.io.read_file(image_path)
            
            # Decode the image
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            
            # Ensure the image has a shape
            image.set_shape([None, None, 3])
            
            # Resize the image
            image = tf.image.resize(image, (256, 256))
            
            # Convert to float32 and normalize to [0, 1]
            image = tf.cast(image, tf.float32) / 255.0
            
            # Normalize to [-1, 1]
            image = ImageUtils.normalize(image)
            
            # Add batch dimension
            image = tf.expand_dims(image, 0)
            
            logger.info(f"Successfully preprocessed image from {image_path}")
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image from {image_path}: {str(e)}")
            raise
