import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from math import pi, cos

# Hyperparameters
BATCH_SIZE = 32
LATENT_DIM = 100 #maybe increase ??
GP_WEIGHT = 10
N_CRITIC = 5 # maybe lower ??
INIT_LR = 5e-3  # Initial learning rate, we can increase later as well
MIN_LR = 1e-5   # Minimum learning rate
EPOCHS = 500 #can edit later
IMAGE_SIZE = 256

# Weight initializer
weight_init = initializers.RandomNormal(mean=0.0, stddev=0.02)

# Cosine learning rate decay
def cosine_decay(epoch):
    decay = 0.5 * (1 + cos(pi * epoch / EPOCHS))
    return MIN_LR + (INIT_LR - MIN_LR) * decay

def build_generator():
    inputs = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(4*4*512, use_bias=False, kernel_initializer=weight_init)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((4, 4, 512))(x)

    for filters in [256, 128, 64, 32, 16]:
        x = layers.Conv2DTranspose(filters, 5, strides=2, padding='same', use_bias=False, kernel_initializer=weight_init)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    # Output layer for grayscale image
    outputs = layers.Conv2DTranspose(1, 5, strides=2, padding='same', use_bias=False,
                                   kernel_initializer=weight_init, activation='tanh')(x)
    return models.Model(inputs, outputs)

def build_discriminator():
    inputs = layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 1])
    x = inputs
    for filters in [64, 128, 256, 512, 512, 512]:
        x = layers.Conv2D(filters, 5, strides=2, padding='same', kernel_initializer=weight_init)(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs, outputs)

def gradient_penalty(discriminator, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    interpolated = real_images * alpha + fake_images * (1 - alpha)
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2,3]) + 1e-12)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(min(16, predictions.shape[0])):
        plt.subplot(4, 4, i+1)
        img = (predictions[i, :, :, 0].numpy() + 1) / 2.0  # Convert to [0,1]
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Epoch {epoch+1}')
    plt.tight_layout()
    plt.show()

def display_real_images(dataset, num_images=16):
    plt.figure(figsize=(4, 4))
    for i, image in enumerate(dataset.take(num_images)):
        plt.subplot(4, 4, i+1)
        img = (image.numpy() + 1) / 2.0  # Convert from [-1,1] to [0,1]
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.axis('off')
    plt.suptitle('Real Training Images')
    plt.tight_layout()
    #plt.show()

@tf.function
def train_step(real_images):
    current_batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([current_batch_size, LATENT_DIM])
    d_losses = []

    for _ in range(N_CRITIC):
        with tf.GradientTape() as d_tape:
            fake_images = generator(noise, training=True)
            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(fake_images, training=True)
            d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gp = gradient_penalty(discriminator, real_images, fake_images)
            d_loss_total = d_loss + GP_WEIGHT * gp
        d_grad = d_tape.gradient(d_loss_total, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))
        d_losses.append(d_loss_total)

    with tf.GradientTape() as g_tape:
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images, training=True)
        g_loss = -tf.reduce_mean(fake_output)
    g_grad = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))
    return tf.reduce_mean(d_losses), g_loss

# Load image files
image_dir = '/Users/travis/Desktop/Projects/AI Summer 25/AI-Guided-Metasurfaces-for-Controlling-Thermal-Radiation/Data/b_squares'
image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_dir) for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(image_files)} images.")

# Shuffle and split 80/20
random.shuffle(image_files)
split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
test_files = image_files[split_idx:]
print(f"Train images: {len(train_files)}, Test images: {len(test_files)}")

def load_and_preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=1)
    img.set_shape([None, None, 1])
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img

# Build datasets
train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(test_files)
test_dataset = test_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Display some real images before training starts
print("\nDisplaying sample real images from training set:")
display_real_images(train_dataset.unbatch())

# Optimizers
d_optimizer = optimizers.Adam(learning_rate=INIT_LR, beta_1=0.5, beta_2=0.9)
g_optimizer = optimizers.Adam(learning_rate=INIT_LR, beta_1=0.5, beta_2=0.9)

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Static noise vector for monitoring progress
test_noise = tf.random.normal([16, LATENT_DIM])

for epoch in range(EPOCHS):
    # Update learning rates with cosine decay
    current_lr = cosine_decay(epoch)
    d_optimizer.learning_rate.assign(current_lr)
    g_optimizer.learning_rate.assign(current_lr)

    epoch_d_loss = []
    epoch_g_loss = []

    for batch in train_dataset:
        d_loss, g_loss = train_step(batch)
        epoch_d_loss.append(d_loss)
        epoch_g_loss.append(g_loss)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Discriminator loss: {np.mean(epoch_d_loss):.4f}")
    print(f"Generator loss: {np.mean(epoch_g_loss):.4f}")
    print(f"Learning rate: {current_lr:.2e}")

    # Generate sample images every 5 epochs
    if (epoch + 1) % 1 == 0 or epoch == 0:
        generate_and_save_images(generator, epoch, test_noise)

    # Save models every 50 epochs
    if (epoch + 1) % 100000000000 == 0:
        generator.save_weights(f'generator_epoch_{epoch+1}.h5')
        discriminator.save_weights(f'discriminator_epoch_{epoch+1}.h5')
        print(f"Saved model weights at epoch {epoch+1}")