#BCE Loss + Exponential Learning Rate Decay
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers, losses
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Helper sampling function
def sample_beta1():
    return np.random.uniform(0, 1)

# Hyperparameters
class DCGANConfig:
    def __init__(self):
        self.LATENT_DIM = 128
        self.IMAGE_SIZE = 256
        self.IMAGE_CHANNELS = 1

        self.INITIAL_LR = 9e-2
        self.FINAL_LR = 1e-5
        self.DECAY_RATE = 0.95
        self.DECAY_STEPS = 10
        self.BETA1 = sample_beta1()

        self.BATCH_SIZE = 16
        self.EPOCHS = 1000
        self.N_CRITIC = 1

        self.DATA_PATH = '/Users/travis/Downloads/Data/b_squares'
        self.VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')
        self.TRAIN_TEST_SPLIT = 0.8

        self.OUTPUT_DIR = './dcgan_output'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

class DCGANBuilder:
    @staticmethod
    def build_generator(config):
        model = models.Sequential([
            layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(config.LATENT_DIM,)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((4, 4, 1024)),

            layers.Conv2DTranspose(512, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2DTranspose(256, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2DTranspose(32, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2DTranspose(config.IMAGE_CHANNELS, 5, strides=2, padding='same', use_bias=False, activation='tanh')
        ])
        return model

    @staticmethod
    def build_discriminator(config):
        model = models.Sequential([
            layers.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNELS)),
            layers.Conv2D(16, 5, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(32, 5, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(64, 5, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(128, 5, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(256, 5, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Conv2D(512, 5, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        return model

class CustomImageDataset:
    def __init__(self, config):
        self.config = config
        self.image_files = self._find_image_files()
        self._split_dataset()

    def _find_image_files(self):
        image_files = [
            os.path.join(dp, f)
            for dp, _, filenames in os.walk(self.config.DATA_PATH)
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
        img = tf.image.decode_image(img, channels=self.config.IMAGE_CHANNELS)
        img.set_shape([None, None, self.config.IMAGE_CHANNELS])
        img = tf.image.resize(img, [self.config.IMAGE_SIZE, self.config.IMAGE_SIZE])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img

    def get_train_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.train_files)
        dataset = dataset.map(self._load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000).batch(self.config.BATCH_SIZE)
        return dataset.prefetch(tf.data.AUTOTUNE)

class DCGANTrainer:
    def __init__(self, config, generator, discriminator, dataset):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset

        self.cross_entropy = losses.BinaryCrossentropy(from_logits=False)

        self.g_optimizer = optimizers.Adam(learning_rate=config.INITIAL_LR, beta_1=config.BETA1)
        self.d_optimizer = optimizers.Adam(learning_rate=config.INITIAL_LR, beta_1=config.BETA1)

        self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')
        self.d_loss_metric = tf.keras.metrics.Mean(name='d_loss')

        self.fixed_noise = tf.random.normal([16, config.LATENT_DIM])

        print("\n=== DCGAN Config ===")
        print(f"Initial LR: {config.INITIAL_LR:.1e}, Final LR: {config.FINAL_LR:.1e}, β1: {config.BETA1:.2f}")
        print("=====================\n")

    def lr_schedule(self, epoch):
        lr = self.config.INITIAL_LR * (self.config.DECAY_RATE ** (epoch / self.config.DECAY_STEPS))
        return max(lr, self.config.FINAL_LR)

    def discriminator_loss(self, real_output, fake_output):
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)
        real_loss = self.cross_entropy(real_labels, real_output)
        fake_loss = self.cross_entropy(fake_labels, fake_output)
        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.config.LATENT_DIM])

        # Train Discriminator
        with tf.GradientTape() as d_tape:
            fake_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            d_loss = self.discriminator_loss(real_output, fake_output)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as g_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            g_loss = self.generator_loss(fake_output)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return d_loss, g_loss

    def generate_samples(self, epoch, lr):
        predictions = self.generator(self.fixed_noise, training=False)
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            img = (predictions[i, :, :, 0].numpy() + 1) / 2.0
            plt.imshow(img, cmap='gray')
            plt.axis('off')

        plt.suptitle(f'Epoch {epoch + 1}\nLR: {lr:.1e} | β1: {self.config.BETA1:.2f}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, f'samples_epoch_{epoch + 1:03d}.png'))
        plt.close()

    def train(self):
        train_dataset = self.dataset.get_train_dataset()

        for epoch in range(self.config.EPOCHS):
            self.g_loss_metric.reset_state()
            self.d_loss_metric.reset_state()

            lr = self.lr_schedule(epoch)
            self.g_optimizer.learning_rate.assign(lr)
            self.d_optimizer.learning_rate.assign(lr)

            print(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}")
            for batch in train_dataset:
                d_loss, g_loss = self.train_step(batch)
                self.d_loss_metric.update_state(d_loss)
                self.g_loss_metric.update_state(g_loss)

            print(f"D_loss: {self.d_loss_metric.result():.4f} | G_loss: {self.g_loss_metric.result():.4f}")
            self.generate_samples(epoch, lr)

            if (epoch + 1) % 10 == 0:
                try:
                    save_path = os.path.join(self.config.OUTPUT_DIR, f'generator_epoch_{epoch + 1:03d}.h5')
                    self.generator.save(save_path)
                    print(f"Generator model saved to {save_path}")
                except Exception as e:
                    print(f"Error saving generator model: {e}")

if __name__ == "__main__":
    config = DCGANConfig()
    generator = DCGANBuilder.build_generator(config)
    discriminator = DCGANBuilder.build_discriminator(config)
    dataset = CustomImageDataset(config)
    trainer = DCGANTrainer(config, generator, discriminator, dataset)
    trainer.train()
