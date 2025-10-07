# DCGAN with Cosine Learning Rate Decay and 32x32 Inputs
# Learning rate decay is too slow
# Updated 6/9/25 

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Helper function to sample beta1 for Adam optimizer
def sample_beta1():
    return np.random.uniform(0, 1)

# Hyperparameters and configuration
class DCGANConfig:
    def __init__(self):
        self.LATENT_DIM = 128
        self.IMAGE_SIZE = 32
        self.IMAGE_CHANNELS = 1

        self.INITIAL_LR = 1e-3
        self.FINAL_LR = 1e-5
        self.BETA1 = sample_beta1()

        self.BATCH_SIZE = 16
        self.EPOCHS = 50
        self.N_CRITIC = 10

        self.DATA_PATH = '/Users/travis/Desktop/Projects/AI Summer 25/AI-Guided-Metasurfaces-for-Controlling-Thermal-Radiation/Data/b_squares'
        self.VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')
        self.TRAIN_TEST_SPLIT = 0.8

        self.OUTPUT_DIR = './dcgan_output'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

# Build the Generator model
class DCGANBuilder:
    @staticmethod
    def build_generator(config):
        model = models.Sequential([
            # Starting from latent vector to 4x4x1024 feature map
            layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(config.LATENT_DIM,)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((4, 4, 1024)),

            # Upsample to 8x8
            layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Upsample to 16x16
            layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Upsample to 32x32
            layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Output layer: 32x32xchannels with tanh activation
            layers.Conv2DTranspose(config.IMAGE_CHANNELS, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')
        ])
        return model

    @staticmethod
    def build_discriminator(config):
        model = models.Sequential([
            layers.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS)),

            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Flatten(),  # output shape will be (batch_size, 2*2*512) = (batch_size, 2048)

            layers.Dense(1)  # output: real/fake score
        ])

        # Build and print summary for debugging shape correctness
        model.build(input_shape=(None, config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS))
        print("\nDiscriminator Model Summary:")
        model.summary()

        return model

# Dataset loader and preprocessor
class CustomImageDataset:
    def __init__(self, config):
        self.config = config
        self.image_files = self._find_image_files()
        self._split_dataset()

    def _find_image_files(self):
        image_files = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(self.config.DATA_PATH)
            for f in filenames
            if f.lower().endswith(self.config.VALID_EXTENSIONS)
        ]
        print(f"Found {len(image_files)} images.")
        return image_files

    def _split_dataset(self):
        random.shuffle(self.image_files)
        split_idx = int(len(self.image_files) * self.config.TRAIN_TEST_SPLIT)
        self.train_files = self.image_files[:split_idx]
        self.test_files = self.image_files[split_idx:]
        print(f"Train images: {len(self.train_files)}, Test images: {len(self.test_files)}")

    def _load_and_preprocess_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=self.config.IMAGE_CHANNELS, expand_animations=False)
        img.set_shape([None, None, self.config.IMAGE_CHANNELS])
        img = tf.image.resize(img, [self.config.IMAGE_SIZE, self.config.IMAGE_SIZE])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
        return img

    def get_train_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.train_files)
        dataset = dataset.map(self._load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000).batch(self.config.BATCH_SIZE)
        return dataset.prefetch(tf.data.AUTOTUNE)

# Trainer class for WGAN-GP with cosine learning rate decay
class WGANGPTrainer:
    def __init__(self, config, generator, discriminator, dataset):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset

        total_steps = config.EPOCHS
        self.cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.INITIAL_LR,
            decay_steps=total_steps,
            alpha=config.FINAL_LR / config.INITIAL_LR
        )

        self.g_optimizer = optimizers.Adam(learning_rate=self.cosine_schedule, beta_1=config.BETA1)
        self.d_optimizer = optimizers.Adam(learning_rate=self.cosine_schedule, beta_1=config.BETA1)

        self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')
        self.d_loss_metric = tf.keras.metrics.Mean(name='d_loss')

        self.fixed_noise = tf.random.normal([16, config.LATENT_DIM])

        print("\n=== DCGAN Config ===")
        print(f"Initial LR: {config.INITIAL_LR:.1e}, Final LR: {config.FINAL_LR:.1e}, β1: {config.BETA1:.3f}")
        print("=====================\n")

    def gradient_penalty(self, real_images, fake_images):
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
        interpolated = real_images * alpha + fake_images * (1 - alpha)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
        return tf.reduce_mean((norm - 1.0) ** 2)

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.config.LATENT_DIM])

        d_losses = []
        for _ in range(self.config.N_CRITIC):
            with tf.GradientTape() as d_tape:
                fake_images = self.generator(noise, training=True)
                real_output = self.discriminator(real_images, training=True)
                fake_output = self.discriminator(fake_images, training=True)
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                gp = self.gradient_penalty(real_images, fake_images)
                d_loss_total = d_loss + 10 * gp

            d_grad = d_tape.gradient(d_loss_total, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
            d_losses.append(d_loss_total)

        with tf.GradientTape() as g_tape:
            fake_images = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_output)

        g_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        return tf.reduce_mean(d_losses), g_loss

    def generate_samples(self, epoch, lr):
        predictions = self.generator(self.fixed_noise, training=False)
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            img = (predictions[i, :, :, 0].numpy() + 1) / 2.0
            plt.imshow(img, cmap='gray')
            plt.axis('off')

        plt.suptitle(f'Epoch {epoch + 1}\nLR: {lr:.1e} | β1: {self.config.BETA1:.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, f'samples_epoch_{epoch + 1:03d}.png'))
        plt.close()

    def train(self):
        train_dataset = self.dataset.get_train_dataset()

        for epoch in range(self.config.EPOCHS):
            self.g_loss_metric.reset_state()
            self.d_loss_metric.reset_state()

            lr = self.cosine_schedule(epoch).numpy()

            print(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}")
            for batch in train_dataset:
                d_loss, g_loss = self.train_step(batch)
                self.d_loss_metric.update_state(d_loss)
                self.g_loss_metric.update_state(g_loss)

            print(f"D_loss: {self.d_loss_metric.result():.4f} | G_loss: {self.g_loss_metric.result():.4f}")
            self.generate_samples(epoch, lr)

            if (epoch + 1) % 10 == 0:
                save_path = os.path.join(self.config.OUTPUT_DIR, f'generator_epoch_{epoch + 1:03d}.h5')
                try:
                    self.generator.save(save_path)
                    print(f"Generator model saved to {save_path}")
                except Exception as e:
                    print(f"Error saving generator model: {e}")

if __name__ == "__main__":
    config = DCGANConfig()
    generator = DCGANBuilder.build_generator(config)
    discriminator = DCGANBuilder.build_discriminator(config)
    dataset = CustomImageDataset(config)
    trainer = WGANGPTrainer(config, generator, discriminator, dataset)
    trainer.train()
