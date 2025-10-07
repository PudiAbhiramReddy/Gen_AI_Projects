import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from google.colab import drive
drive.mount('/content/drive')


# --- 0. Setup and Hyperparameters ---
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Training will be on CPU.")


# IMPORTANT: Set this to the PARENT folder containing your category subfolders
DATAROOT = "/content/drive/MyDrive/MetaSurfaces_Final1/Data"  # e.g., "/content/My_Metasurface_Data"
IMAGE_SIZE = 64
CHANNELS = 1  # Grayscale
BATCH_SIZE = 64
BUFFER_SIZE = 10000
NOISE_DIM = 100
NGF = 64  # Number of generator features
NDF = 64  # Number of critic (discriminator) features
NUM_EPOCHS = 300 # WGAN-GP might need more epochs, or show good results sooner
LR_G = 0.0001 # Learning rates often smaller for WGAN-GP
LR_C = 0.0001 # Critic learning rate
BETA_1_ADAM = 0.0 # Common beta1 for Adam in WGAN-GP
BETA_2_ADAM = 0.9 # Common beta2 for Adam in WGAN-GP
N_CRITIC = 5      # Number of critic updates per generator update
GP_WEIGHT = 10.0  # Gradient penalty weight (lambda)

# Create output directories
OUTPUT_DIR = "/content/drive/MyDrive/wgan_gp_output_metasurfaces_tf2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)

# --- 2. Data Loading and Preprocessing (Same as before) ---
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=CHANNELS)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.BICUBIC)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image

try:
    list_ds = tf.data.Dataset.list_files(str(DATAROOT + '/*/*.png'), shuffle=False) # Add shuffle=False here
    if not list_ds.cardinality().numpy() > 0:
        list_ds = tf.data.Dataset.list_files(str(DATAROOT + '/*/*.jpeg'), shuffle=False)
        if not list_ds.cardinality().numpy() > 0:
            print(f"Error: No image files found in {DATAROOT} with .png or .jpeg extension.")
            exit()

    # Shuffle filenames before mapping for better randomness if BUFFER_SIZE is smaller than dataset
    list_ds = list_ds.shuffle(buffer_size=tf.data.experimental.cardinality(list_ds).numpy())


    train_dataset = list_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE) # drop_remainder for consistent batch sizes

    print(f"Dataset loaded. Number of batches: {len(list(train_dataset))}")
    # Plot some training images (same plotting code as before)
    for image_batch in train_dataset.take(1):
        plt.figure(figsize=(8, 8))
        for i in range(min(16, BATCH_SIZE)):
            plt.subplot(4, 4, i + 1)
            img_to_show = (image_batch[i, :, :, 0].numpy() * 0.5) + 0.5
            plt.imshow(img_to_show, cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Sample Training Images (Resized to {IMAGE_SIZE}x{IMAGE_SIZE})")
        plt.savefig(os.path.join(OUTPUT_DIR, "sample_training_images_wgan_gp.png"))
        plt.show()
        break
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()


# --- 3. Model Definitions ---

# Generator (Can be similar to DCGAN's generator)
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*NGF*8, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((4, 4, NGF * 8)))

    model.add(layers.Conv2DTranspose(NGF * 4, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(NGF * 2, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(NGF, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

generator = make_generator_model()



# Critic (Discriminator for WGAN-GP)
# - No Batch Normalization (typically)
# - No Sigmoid output
def make_critic_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(NDF, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[IMAGE_SIZE, IMAGE_SIZE, CHANNELS]))
    model.add(layers.LeakyReLU(alpha=0.2))
    # No Batch Norm

    model.add(layers.Conv2D(NDF * 2, (5, 5), strides=(2, 2), padding='same'))
    # No Batch Norm (or use LayerNormalization if desired: layers.LayerNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(NDF * 4, (5, 5), strides=(2, 2), padding='same'))
    # No Batch Norm
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(NDF * 8, (5, 5), strides=(2, 2), padding='same'))
    # No Batch Norm
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # Outputs a raw score (logit)
    return model

critic = make_critic_model()

# --- 4. Loss Functions and Optimizers ---

# Optimizers (Adam with specific betas is common for WGAN-GP)
generator_optimizer = tf.keras.optimizers.Adam(LR_G, beta_1=BETA_1_ADAM, beta_2=BETA_2_ADAM)
critic_optimizer = tf.keras.optimizers.Adam(LR_C, beta_1=BETA_1_ADAM, beta_2=BETA_2_ADAM)


# Loss functions for WGAN-GP
def critic_loss_fn(real_output, fake_output):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    return fake_loss - real_loss # We want to maximize real_output - fake_output

def generator_loss_fn(fake_output):
    return -tf.reduce_mean(fake_output) # Generator wants to maximize critic's score on fakes

# Gradient Penalty
@tf.function
def gradient_penalty(real_images, fake_images, batch_size_gp):
    # Random alpha for interpolation
    alpha = tf.random.normal([batch_size_gp, 1, 1, 1], 0.0, 1.0) # Shape for broadcasting
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the critic output for this interpolated image.
        pred = critic(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])) # Sum over H, W, C
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp




# --- Checkpoints ---
checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 critic_optimizer=critic_optimizer, # Renamed
                                 generator=generator,
                                 critic=critic, # Renamed
                                 epoch=tf.Variable(0))



# --- For generating sample images ---
seed = tf.random.normal([16, NOISE_DIM])



# --- 5. Training Step Definition ---
@tf.function
def train_step(real_images_batch):
    current_batch_size = tf.shape(real_images_batch)[0]
    # --- Train Critic N_CRITIC times ---
    for _ in range(N_CRITIC):
        noise_for_critic = tf.random.normal([current_batch_size, NOISE_DIM])
        with tf.GradientTape() as crit_tape:
            generated_images = generator(noise_for_critic, training=True)

            real_output = critic(real_images_batch, training=True)
            fake_output = critic(generated_images, training=True)

            crit_loss_wasserstein = critic_loss_fn(real_output, fake_output)
            gp = gradient_penalty(real_images_batch, generated_images, current_batch_size)
            total_crit_loss = crit_loss_wasserstein + GP_WEIGHT * gp

        gradients_of_critic = crit_tape.gradient(total_crit_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

    # --- Train Generator ---
    noise_for_generator = tf.random.normal([current_batch_size, NOISE_DIM])
    with tf.GradientTape() as gen_tape:
        generated_images_for_G = generator(noise_for_generator, training=True)
        fake_output_for_G = critic(generated_images_for_G, training=True) # Critic assesses G's fakes
        gen_loss = generator_loss_fn(fake_output_for_G)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    # For logging, D(x) is E[critic(real)], D(G(z)) is E[critic(fake)] from the critic's perspective
    # These are raw scores, not probabilities. Higher D(x) and lower D(G(z)) is what critic aims for.
    # G aims to make D(G(z)) higher.
    d_x_score = tf.reduce_mean(real_output) # From last critic update on real
    d_g_z1_score = tf.reduce_mean(fake_output) # From last critic update on fake
    d_g_z2_score = tf.reduce_mean(fake_output_for_G) # From generator's update, G wants this high

    return total_crit_loss, gen_loss, d_x_score, d_g_z1_score, d_g_z2_score


# --- Generate and Save Images Function (Same as before) ---
def generate_and_save_images(model, epoch, test_input, output_dir_img):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img_to_show = (predictions[i, :, :, 0].numpy() * 0.5) + 0.5
        plt.imshow(img_to_show, cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join(output_dir_img, f'image_at_epoch_{epoch:04d}.png'))
    plt.close(fig)



# --- 6. Training Loop (Modified for WGAN-GP logging) ---
def train(dataset, total_epochs, start_epoch):
    G_losses_wgan = []
    C_losses_wgan = [] # Critic losses

    for epoch_idx in range(start_epoch, total_epochs):
        current_epoch_display = epoch_idx + 1
        checkpoint.epoch.assign(current_epoch_display)

        start_time = time.time()
        batch_num = 0
        for image_batch in dataset:
            crit_loss, gen_loss, c_x_score, c_g_z1_score, c_g_z2_score = train_step(image_batch)
            batch_num += 1

            if batch_num % 50 == 0:
                print(f'Epoch {current_epoch_display}/{total_epochs}, Batch {batch_num}/{len(list(dataset))}, '
                      f'Loss_C: {crit_loss:.4f}, Loss_G: {gen_loss:.4f}, '
                      f'C(x): {c_x_score:.4f}, C(G(z)): {c_g_z1_score:.4f} / {c_g_z2_score:.4f}')

            G_losses_wgan.append(gen_loss.numpy())
            C_losses_wgan.append(crit_loss.numpy())

        generate_and_save_images(generator, current_epoch_display, seed, os.path.join(OUTPUT_DIR, "images"))

        if (current_epoch_display % 10 == 0) or (current_epoch_display == total_epochs):
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"Saved checkpoint for epoch {current_epoch_display}")

        print(f'Time for epoch {current_epoch_display} is {time.time()-start_time:.2f} sec')
    return G_losses_wgan, C_losses_wgan


# --- Attempt to Restore Checkpoint (Same as before) ---
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
initial_epoch = 0
if latest_ckpt:
    try:
        status = checkpoint.restore(latest_ckpt)
        initial_epoch = checkpoint.epoch.numpy()
        print(f"Checkpoint restored from {latest_ckpt}. Resuming from epoch {initial_epoch + 1}.")
    except Exception as e:
        print(f"Could not fully restore epoch from checkpoint: {e}. Model/Optimizer weights might be restored.")
        # MANUALLY SET THE STARTING EPOCH IF NEEDED, e.g., if previous run was DCGAN
        # initial_epoch = 0 # Or relevant epoch if resuming a previous WGAN-GP run
        print(f"Starting WGAN-GP training. Previous DCGAN checkpoints are not compatible.")
else:
    print("No WGAN-GP checkpoint found. Starting training from scratch.")

# --- Call the training function ---
print("Starting WGAN-GP Training...")
G_losses, C_losses = train(train_dataset, NUM_EPOCHS, initial_epoch)
print("Training Finished.")

# --- Plotting Results ---
plt.figure(figsize=(10,5))
plt.title("Generator and Critic Loss During WGAN-GP Training (TensorFlow)")
plt.plot(G_losses,label="G")
plt.plot(C_losses,label="C (Critic)")
plt.xlabel("iterations (batches)")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve_wgan_gp_tf.png"))
plt.show()

print(f"Outputs saved in directory: {OUTPUT_DIR}")
