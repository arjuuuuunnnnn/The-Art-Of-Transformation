import os
import tensorflow as tf
from dotenv import load_dotenv
from src import logger
from src.Transform.utils import ImageUtils
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()

def download_and_prepare_dataset(kaggle_dataset, data_dir='data'):
    os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

    api = KaggleApi()
    api.authenticate()

    os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Downloading dataset: {kaggle_dataset}")
    api.dataset_download_files(kaggle_dataset, path=data_dir, unzip=True)

    image_dir = os.path.join(data_dir, 'photos')
    sketch_dir = os.path.join(data_dir, 'sketches')

    def load_and_preprocess_image(image_path, sketch_path):
        image = ImageUtils.preprocess_image(image_path)
        image = tf.squeeze(image, axis=0)  # Remove batch dimension

        sketch = tf.io.read_file(sketch_path)
        sketch = tf.image.decode_jpeg(sketch, channels=1)
        sketch = tf.image.resize(sketch, [256, 256])
        sketch = ImageUtils.normalize(tf.cast(sketch, tf.float32))

        return image, sketch

    # Match images and sketches by filenames
    image_paths = sorted(tf.io.gfile.glob(os.path.join(image_dir, '*.jpg')))
    sketch_paths = sorted(tf.io.gfile.glob(os.path.join(sketch_dir, '*.jpg')))

    # Create TensorFlow Dataset from the lists
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    sketch_dataset = tf.data.Dataset.from_tensor_slices(sketch_paths)

    dataset = tf.data.Dataset.zip((image_dataset, sketch_dataset))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return dataset

