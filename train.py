import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import gc

from src.Transform.encoder import Encoder
from src.Transform.generator import Generator
from src.Transform.discriminator import Discriminator
from src.Transform.utils import ImageUtils
from src import logger
from dataset import download_and_prepare_dataset

# Set the mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define a function to normalize images
@tf.function
def normalize(arr):
    arr_min = tf.reduce_min(arr)
    arr_max = tf.reduce_max(arr)
    arr_range = arr_max - arr_min
    scaled = (arr - arr_min) / arr_range
    return -1 + (scaled * 2)

class Trainer:
    def __init__(self):
        self.encoder = Encoder()
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.generator_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
        self.discriminator_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
        self.encoder_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

        self.binary_cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        self.mae = keras.losses.MeanAbsoluteError()

        
        self.checkpoint_prefix = "checkpoints/ckpt"
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.ckpt = tf.train.Checkpoint(generator=self.generator,
                                        discriminator=self.discriminator,
                                        encoder=self.encoder,
                                        generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        encoder_optimizer=self.encoder_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=3)

        
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            logger.info('Latest checkpoint restored!!')

    @tf.function
    def train_step(self, real_images, real_sketches):
        with tf.GradientTape(persistent=True) as tape:
            
            latent_code = self.encoder(real_images)

            
            generated_sketches = self.generator(latent_code)

            
            real_sketches = tf.ensure_shape(real_sketches, (None, 256, 256, None))

            if real_sketches.shape[-1] == 1:
                real_sketches = tf.repeat(real_sketches, 3, axis=-1)
            real_sketches = tf.ensure_shape(real_sketches, (None, 256, 256, 3))

            generated_sketches = tf.ensure_shape(generated_sketches, (None, 256, 256, 3))
            

            real_output = self.discriminator(real_sketches)
            fake_output = self.discriminator(generated_sketches)

            epsilon = 1e-12
            disc_loss = self.binary_cross_entropy(tf.ones_like(real_output), real_output + epsilon) + \
                        self.binary_cross_entropy(tf.zeros_like(fake_output), fake_output + epsilon)

            gen_loss = self.binary_cross_entropy(tf.ones_like(fake_output), fake_output + epsilon)
            reconstruction_loss = self.mae(real_sketches, generated_sketches)
            total_loss = gen_loss + reconstruction_loss

        gradients_of_discriminator = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_of_generator = tape.gradient(total_loss, self.generator.trainable_variables)
        gradients_of_encoder = tape.gradient(total_loss, self.encoder.trainable_variables)

        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.encoder_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_variables))

        return {"disc_loss": disc_loss, "gen_loss": gen_loss, "reconstruction_loss": reconstruction_loss}

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")

            epoch_disc_loss = 0
            epoch_gen_loss = 0
            epoch_recon_loss = 0
            num_batches = 0

            for real_images_batch, real_sketches_batch in dataset:
                losses = self.train_step(real_images_batch, real_sketches_batch)
                epoch_disc_loss += losses['disc_loss']
                epoch_gen_loss += losses['gen_loss']
                epoch_recon_loss += losses['reconstruction_loss']
                num_batches += 1

            avg_disc_loss = epoch_disc_loss / num_batches
            avg_gen_loss = epoch_gen_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches

            logger.info(f"Epoch {epoch + 1} - Discriminator loss: {avg_disc_loss:.4f}, Generator loss: {avg_gen_loss:.4f}, Reconstruction loss: {avg_recon_loss:.4f}")

            
            gc.collect()
            tf.keras.backend.clear_session()

            
            self.ckpt_manager.save()

        
        self.encoder.save('encoder_model')
        self.generator.save('generator_model')
        self.discriminator.save('discriminator_model')
        logger.info("Models saved after training")

if __name__ == "__main__":
    kaggle_dataset = "arbazkhan971/cuhk-face-sketch-database-cufs"
    data_dir = 'data'
    
    train_dataset = download_and_prepare_dataset(kaggle_dataset, data_dir=data_dir)
    
    trainer = Trainer()
    epochs = 1
    trainer.train(train_dataset, epochs)

