# -*- coding: utf-8 -*-
# Make it to where each absorption test case can generate up to 5 images which produce similar responses, and rank them by similarity in responses, such as 99% accuracy, 94% accuracy, etc . 
# Need to figure out an accuracy function which helps us determine which is the most accurate. 
# Also need to make a generator which takes only the absorption values as an input and is able to generate a image.
# This is our best working version. We will stick with Wgan instead of DC gan, as DC gan significantly underperformed in comparison to Wgan.
# Key things: Need to make it so that we have all graphs output for everything we need, structured based on image, and some graphs for overall performance. Also need to improve convergance time, currently takes >2000 epochs to converge.
"""AI-Guided Metasurface Design: Conditional WGAN-GP with Simulator (Refined)"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import matplotlib
import random

# For non-interactive plotting (e.g., when running on a server without display)
matplotlib.use('Agg')

# --- 0. Device Configuration ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Hyperparameters (Combined and Adjusted for Conditional WGAN-GP & Flexible Image Sizing) ---

# --- Data Paths (EDIT THESE TO YOUR FILE LOCATIONS) ---
# This is the folder containing your image files (e.g., circle_0000.png)
IMAGE_FOLDER_PATH = "/home/fu/travis/Projects/Metasurfaces/AI-Guided-Metasurfaces-for-Controlling-Thermal-Radiation/Data/Data_Generated_Images/a_circles"
# Path to your training absorbance metadata CSV file
TRAIN_METADATA_FILE = "/home/fu/travis/Projects/Metasurfaces/AI-Guided-Metasurfaces-for-Controlling-Thermal-Radiation/metasurface_absorbance_updated.csv"
# Path to your testing absorbance metadata CSV file
TEST_METADATA_FILE = "/home/fu/travis/Projects/Metasurfaces/AI-Guided-Metasurfaces-for-Controlling-Thermal-Radiation/metasurface_absorbance_test_updated.csv"

# --- Image and Data Dimensions ---
# GAN operates at this size for training. Change to PRODUCTION_GAN_IMAGE_SIZE for final training.
CURRENT_GAN_IMAGE_SIZE = 64
# Uncomment and change CURRENT_GAN_IMAGE_SIZE to this value when ready for full resolution training.
# PRODUCTION_GAN_IMAGE_SIZE = 256

# Simulator always operates at this fixed size. Generated images will be resized to this.
SIMULATOR_IMAGE_SIZE = 64

CHANNELS = 1 # Grayscale images
NUM_ANGLES = 15 # Output dimensions for absorbance spectrum (also used as conditioning input size)

# --- GAN Model Parameters ---
LATENT_DIM = 128 # Size of the latent z vector (NOISE_DIM from WGAN-GP) Increase to 256 or 512 as test.
GF = 64 # Generator feature map depth (NGF from WGAN-GP) - Base feature multiplier
DF = 64 # Critic feature map depth (NDF from WGAN-GP) - Base feature multiplier

# --- Simulator Model Parameters ---
SIM_NDF = 64 # Simulator feature map depth (NDF from original simulator guideline)

# --- Training Parameters ---
BATCH_SIZE = 32 # Higher batch size should make our code run better, but takes a lot of processing power. If able, 128 or 256 should be best.
LEARNING_RATE_SIM = 0.0002 # Simulator learning rate

NUM_EPOCHS_SIMULATOR = 100 # Epochs for pre-training the simulator
NUM_EPOCHS_GAN = 3000 # Epochs for GAN training (from original WGAN-GP script)

WORKERS = 2 # Number of workers for dataloader

# --- WGAN-GP Specific Parameters ---
N_CRITIC = 5      # Number of critic updates per generator update
GP_WEIGHT = 10.0  # Gradient penalty weight
BETA1_ADAM = 0.0 # WGAN-GP typical beta1 for Adam (from original WGAN-GP script)
BETA2_ADAM = 0.9 # WGAN-GP typical beta2 for Adam (from original WGAN-GP script)

# --- Learning Rate and Optimizer Adjustments (from original WGAN-GP script) ---
INITIAL_LR_G = 0.0002 # Initial LR for cosine annealing for Generator
INITIAL_LR_C = 0.0002 # Initial LR for cosine annealing for Critic

# Loss weighting parameter for Generator (from the paper: L_G = L_C + λ * L_S)
LAMBDA_SIM_LOSS = 0.0001 # λ from the paper, balancing simulator and critic loss for G
# You can adjust this to change emphasis:
# - Smaller LAMBDA_SIM_LOSS (e.g., 0.0001): Stronger emphasis on image realism (from Critic)
# - Larger LAMBDA_SIM_LOSS (e.g., 1.0): Stronger emphasis on pixel accuracy/property matching (from Simulator)

# --- Output Directories ---
OUTPUT_DIR_BASE = f"wgan_gp_conditional_{CURRENT_GAN_IMAGE_SIZE}x{CURRENT_GAN_IMAGE_SIZE}_output" # Base output directory
OUTPUT_DIR_SIM = os.path.join(OUTPUT_DIR_BASE, "simulator_output")
OUTPUT_DIR_GAN = os.path.join(OUTPUT_DIR_BASE, "gan_output")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR_SIM, exist_ok=True)
os.makedirs(OUTPUT_DIR_GAN, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR_GAN, "checkpoints"), exist_ok=True) # For GAN checkpoints

# --- Weight Initializers (DCGAN-style) ---
def weights_init(m):
    """
    Initializes weights of convolutional, batch normalization, and linear layers
    with a normal distribution (mean=0.0, std=0.02) and biases to zero.
    This is a common practice in DCGANs and WGANs.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # Covers Conv2d and ConvTranspose2d
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) # Gamma (weight)
        nn.init.constant_(m.bias.data, 0)        # Beta (bias)
    elif classname.find('Linear') != -1: # For Dense layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# --- 2. Custom Dataset (Shared for both Simulator and GAN) ---
class SingleFolderMetasurfaceDataset(Dataset):
    """
    Custom Dataset class for loading metasurface images and their
    corresponding absorbance spectra from a folder and a metadata CSV.
    """
    def __init__(self, metadata_file, image_folder_path, transform=None):
        try:
            self.metadata_df = pd.read_csv(metadata_file)
        except FileNotFoundError:
            print(f"Error: Metadata file not found at {metadata_file}")
            raise
        self.image_folder_path = image_folder_path
        self.transform = transform
        # Ensure the metadata file has enough columns for filename + absorbance values
        if len(self.metadata_df.columns) < NUM_ANGLES + 1:
            raise ValueError(f"CSV file must have at least {NUM_ANGLES + 1} columns (filename + {NUM_ANGLES} absorbances)")
        # Identify columns containing absorbance data
        self.absorbance_cols = self.metadata_df.columns[1:NUM_ANGLES+1].tolist()

        # Determine the column containing image filenames
        if 'Image Name' not in self.metadata_df.columns and self.metadata_df.columns[0].lower() != 'image name':
            if 'filename' in self.metadata_df.columns:
                self.filename_col = 'filename'
            else:
                print("Warning: 'Image Name' column not found. Assuming first column is filename.")
                self.filename_col = self.metadata_df.columns[0]
        else:
            self.filename_col = 'Image Name'

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding absorbance vector at the given index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_filename_in_csv = self.metadata_df.iloc[idx][self.filename_col]
        img_full_path = os.path.join(self.image_folder_path, img_filename_in_csv)
        try:
            image = Image.open(img_full_path).convert('L') # Convert to grayscale
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_full_path} (referenced in {self.metadata_df.iloc[idx][self.filename_col]})")
            raise FileNotFoundError(f"Image file not found: {img_full_path}")

        # Extract absorbance values and convert to float32 tensor
        absorbance_vector = self.metadata_df.iloc[idx][self.absorbance_cols].values.astype(np.float32)
        absorbance_tensor = torch.from_numpy(absorbance_vector)

        if self.transform:
            image = self.transform(image)
        return image, absorbance_tensor

# Define image transformations for the GAN, dynamically set by CURRENT_GAN_IMAGE_SIZE
image_transforms_gan = transforms.Compose([
    # Original images in dataset are assumed to be 256x256. Resize to CURRENT_GAN_IMAGE_SIZE for training.
    transforms.Resize(CURRENT_GAN_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(CURRENT_GAN_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS) # Normalize images to [-1, 1]
])

# Define transformations for feeding generated images to simulator (e.g., 256x256 or 64x64 to 64x64)
resize_for_simulator = transforms.Resize(SIMULATOR_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC)


# Setup Datasets and DataLoaders
try:
    train_dataset = SingleFolderMetasurfaceDataset(metadata_file=TRAIN_METADATA_FILE,
                                                       image_folder_path=IMAGE_FOLDER_PATH,
                                                       transform=image_transforms_gan) # Use GAN's image size
    if len(train_dataset) == 0: raise ValueError("Training dataset is empty.")
    # drop_last=True is crucial for WGAN-GP to ensure consistent batch sizes for gradient penalty
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS,
                                    drop_last=True)
    print(f"Total training images: {len(train_dataset)}")
    num_batches_per_epoch = len(train_dataloader)
    DECAY_STEPS_TOTAL = num_batches_per_epoch * NUM_EPOCHS_GAN # T_max for CosineAnnealingLR for GAN

    test_dataloader = None
    if os.path.exists(TEST_METADATA_FILE):
        test_dataset = SingleFolderMetasurfaceDataset(metadata_file=TEST_METADATA_FILE,
                                                          image_folder_path=IMAGE_FOLDER_PATH,
                                                          transform=image_transforms_gan) # Use GAN's image size
        if len(test_dataset) > 0:
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, drop_last=False)
            print(f"Total test images: {len(test_dataset)}")
        else:
            print("Warning: Test CSV loaded but dataset is empty. Test dataloader will not be used.")
            test_dataloader = None
    else:
        print(f"Warning: Test metadata file '{TEST_METADATA_FILE}' not found. Test dataloader will not be used.")

    # Display sample training images and absorbance vector
    if len(train_dataset) > 0:
        sample_batch_img, sample_batch_abs = next(iter(train_dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(f"Sample Training Images (Resized to {CURRENT_GAN_IMAGE_SIZE}x{CURRENT_GAN_IMAGE_SIZE})")
        plt.imshow(np.transpose(vutils.make_grid(sample_batch_img.to(device)[:min(16, BATCH_SIZE)], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig(os.path.join(OUTPUT_DIR_BASE, "sample_training_images_wgan_gp_conditional.png"))
        plt.close() # Close figure to free memory
        print(f"Sample absorbance vector (first in training batch, shape: {sample_batch_abs[0].shape}):\n{sample_batch_abs[0]}")

except Exception as e:
    print(f"Error setting up dataset/dataloader: {e}")
    raise e

# --- 3. Model Definitions ---

# Generator (G) - Modified to be conditional and flexible for image size
class Generator(nn.Module):
    def __init__(self, target_image_size):
        super(Generator, self).__init__()
        # Calculate initial feature map depth based on target_image_size
        # The first ConvTranspose2d will output 4x4.
        # Number of upsampling layers needed = log2(target_image_size / 4)
        num_upsample_layers = int(np.log2(target_image_size / 4))
        # Initial feature multiplier such that the last layer is GF features.
        # GF * (2^num_upsample_layers) features at the 4x4 stage.
        initial_gf_multiplier = 2**(num_upsample_layers)

        layers = []
        # Input is (LATENT_DIM + NUM_ANGLES) x 1 x 1
        layers.append(nn.ConvTranspose2d(LATENT_DIM + NUM_ANGLES, GF * initial_gf_multiplier, 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(GF * initial_gf_multiplier))
        layers.append(nn.ReLU(True))

        current_gf_multiplier = initial_gf_multiplier
        for i in range(num_upsample_layers - 1): # -1 because the last layer handles CHANNELS output
            layers.append(nn.ConvTranspose2d(GF * current_gf_multiplier, GF * (current_gf_multiplier // 2), 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(GF * (current_gf_multiplier // 2)))
            layers.append(nn.ReLU(True))
            current_gf_multiplier //= 2

        # Final layer outputs CHANNELS
        layers.append(nn.ConvTranspose2d(GF * current_gf_multiplier, CHANNELS, 4, 2, 1, bias=False))
        layers.append(nn.Tanh()) # Output images are normalized to [-1, 1]

        self.main = nn.Sequential(*layers)

    def forward(self, noise, absorbance_vector):
        # Expand absorbance_vector to (batch_size, NUM_ANGLES, 1, 1)
        expanded_absorbance = absorbance_vector.unsqueeze(-1).unsqueeze(-1)
        # Concatenate noise and expanded_absorbance along the channel dimension
        combined_input = torch.cat((noise, expanded_absorbance), dim=1)
        return self.main(combined_input)

# Critic (C) - Modified to be conditional and flexible for image size
# No BatchNorm or Sigmoid as per WGAN-GP
class Critic(nn.Module):
    def __init__(self, target_image_size):
        super(Critic, self).__init__()
        # Calculate initial feature map depth based on target_image_size
        # The last Conv2d will output 1x1.
        # Number of downsampling layers needed = log2(target_image_size / 4) + 1 (for the final 4x4 -> 1x1)
        num_downsample_layers = int(np.log2(target_image_size / 4)) + 1 # +1 for the last 4x4 to 1x1 layer

        layers = []
        # Input is (CHANNELS + NUM_ANGLES) x target_image_size x target_image_size
        layers.append(nn.Conv2d(CHANNELS + NUM_ANGLES, DF, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        current_df_multiplier = 1
        for i in range(num_downsample_layers - 2): # -2 because first layer is done, and last layer is 1x1 output
            layers.append(nn.Conv2d(DF * current_df_multiplier, DF * (current_df_multiplier * 2), 4, 2, 1, bias=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_df_multiplier *= 2

        # Final layer outputs a raw score
        layers.append(nn.Conv2d(DF * current_df_multiplier, 1, 4, 1, 0, bias=False)) # Final 4x4 to 1x1 output

        self.main = nn.Sequential(*layers)

    def forward(self, image_input, absorbance_vector):
        # Expand absorbance_vector to (batch_size, NUM_ANGLES, CURRENT_GAN_IMAGE_SIZE, CURRENT_GAN_IMAGE_SIZE)
        expanded_absorbance = absorbance_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, image_input.shape[2], image_input.shape[3])
        # Concatenate image_input and expanded_absorbance along the channel dimension
        combined_input = torch.cat((image_input, expanded_absorbance), dim=1)
        return self.main(combined_input).view(-1, 1).squeeze(1) # Output a scalar score per image

# Simulator (S) - Fixed to 64x64 input
class RobustSimulatorCNN(nn.Module):
    def __init__(self, num_outputs=NUM_ANGLES, ndf=SIM_NDF): # Using SIM_NDF here, operates on 64x64
        super(RobustSimulatorCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Input is CHANNELS x 64 x 64
            nn.Conv2d(CHANNELS, ndf, kernel_size=4, stride=2, padding=1, bias=False), # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), # 32x32 -> 16x16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), # 16x16 -> 8x8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), # 8x8 -> 4x4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Final convolutional layer before flattening (4x4 to 1x1 if kernel=4, stride=1, padding=0 for this layer)
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=0, bias=False), # 4x4 -> 1x1
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        with torch.no_grad():
            dummy_input = torch.randn(1, CHANNELS, SIMULATOR_IMAGE_SIZE, SIMULATOR_IMAGE_SIZE)
            self.cnn_layers.eval() # Ensure eval mode for correct BatchNorm behavior during dummy pass
            cnn_out_size = self.cnn_layers(dummy_input).view(1, -1).size(1)
            self.cnn_layers.train() # Set back to train mode if needed elsewhere in init, or let model.train() handle it

        self.fc_layers = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_outputs),
            nn.Sigmoid() # Output absorbance values between 0 and 1
        )
    def forward(self, image_input):
        cnn_out = self.cnn_layers(image_input)
        flattened = cnn_out.view(cnn_out.size(0), -1)
        output = self.fc_layers(flattened)
        return output

# Instantiate models
netG = Generator(target_image_size=CURRENT_GAN_IMAGE_SIZE).to(device)
netC = Critic(target_image_size=CURRENT_GAN_IMAGE_SIZE).to(device) # Renamed to netC for Critic
simulator_model = RobustSimulatorCNN(num_outputs=NUM_ANGLES, ndf=SIM_NDF).to(device)

# Apply weights_init
netG.apply(weights_init)
netC.apply(weights_init) # Apply to Critic
simulator_model.apply(weights_init)

# --- 4. Loss Functions and Optimizers ---
# For Simulator
criterion_S = nn.MSELoss() # MSE for simulator loss

# Setup Adam optimizers for GAN and Simulator
optimizerC = optim.Adam(netC.parameters(), lr=INITIAL_LR_C, betas=(BETA1_ADAM, BETA2_ADAM))
optimizerG = optim.Adam(netG.parameters(), lr=INITIAL_LR_G, betas=(BETA1_ADAM, BETA2_ADAM))
optimizer_sim = optim.Adam(simulator_model.parameters(), lr=LEARNING_RATE_SIM, betas=(0.5, 0.999)) # betas from original GAN file

# Learning Rate Schedulers for GAN (T_max based on total GAN iterations)
# DECAY_STEPS_TOTAL is calculated after train_dataloader is defined.
# Schedulers step per batch/iteration.
scheduler_G = CosineAnnealingLR(optimizerG, T_max=DECAY_STEPS_TOTAL, eta_min=INITIAL_LR_G*0.001)
scheduler_C = CosineAnnealingLR(optimizerC, T_max=DECAY_STEPS_TOTAL, eta_min=INITIAL_LR_C*0.001)

# Gradient Penalty function for WGAN-GP
def compute_gradient_penalty(critic_net, real_samples, fake_samples, real_absorbances, current_batch_size_gp):
    """
    Computes the gradient penalty for WGAN-GP.
    It interpolates between real and fake samples (images), calculates the critic's output
    for these interpolated images (conditioned on real absorbances), and then computes the L2 norm
    of the gradients with respect to the image interpolates.
    """
    alpha = torch.rand(current_batch_size_gp, 1, 1, 1, device=device) # Uniformly sample for images

    # Interpolate images. real_absorbances are used for conditioning the critic, not interpolated themselves for GP.
    # The GP is on the image input to the critic.
    interpolates_img = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # The critic receives both interpolated image and the original real absorbance (as conditioning)
    # The gradient penalty is calculated with respect to the image input, *given* the absorbance.
    c_interpolates = critic_net(interpolates_img, real_absorbances)
    gradients = torch.autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates_img, # Only calculate gradients wrt image interpolates
        grad_outputs=torch.ones_like(c_interpolates, device=device), # Using ones_like for scalar output
        create_graph=True, # Required for higher-order derivatives (backward on backward)
        retain_graph=True, # Required if graph will be used again (e.g., for subsequent GP calculations in N_CRITIC loop)
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)  # Flatten gradients for norm calculation
    gradient_penalty_val = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty_val

# Fixed noise and fixed absorbance target for Generator visualization and evaluation
# Dynamically get a sample absorbance from the first batch of the training dataloader
# This ensures it's always a valid absorbance from your actual data.
try:
    # Get one batch to sample a fixed absorbance from. Ensure data loader has at least one item.
    sample_batch_for_fixed_abs = next(iter(train_dataloader))
    # Take the first absorbance vector from the batch
    fixed_absorbance_for_gen = sample_batch_for_fixed_abs[1][0].unsqueeze(0).to(device)
    print(f"Using first absorbance vector from training data for fixed generator visualization: {fixed_absorbance_for_gen.cpu().numpy()}")
except StopIteration:
    print("Warning: Training dataloader is empty. Cannot sample fixed absorbance for generator visualization. Using zeros.")
    fixed_absorbance_for_gen = torch.zeros(1, NUM_ANGLES, device=device) # Fallback if dataloader is empty
fixed_noise = torch.randn(min(64, BATCH_SIZE), LATENT_DIM, 1, 1, device=device)
# If batch size is larger than 1, repeat the fixed absorbance for the whole fixed_noise batch
if fixed_noise.shape[0] > 1:
    fixed_absorbance_for_gen = fixed_absorbance_for_gen.repeat(fixed_noise.shape[0], 1)


# --- 5. Pre-train Simulator ---
print("--- Starting Simulator Pre-training ---")
train_losses_sim = []
test_losses_sim = []

for epoch in range(NUM_EPOCHS_SIMULATOR):
    simulator_model.train()
    running_train_loss = 0.0
    epoch_start_time = time.time()

    for i, (images, targets) in enumerate(train_dataloader):
        # Images from train_dataloader are CURRENT_GAN_IMAGE_SIZE.
        # Resize them to SIMULATOR_IMAGE_SIZE (64x64) for the simulator.
        images = images.to(device)
        images_for_sim = resize_for_simulator(images) # Resize for simulator input
        targets = targets.to(device)

        optimizer_sim.zero_grad()
        outputs = simulator_model(images_for_sim)
        loss = criterion_S(outputs, targets)
        loss.backward()
        optimizer_sim.step()
        running_train_loss += loss.item() * images.size(0)

    epoch_train_loss = running_train_loss / len(train_dataset)
    train_losses_sim.append(epoch_train_loss)

    epoch_test_loss_val = float('nan')
    if test_dataloader:
        simulator_model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for images_test, targets_test in test_dataloader:
                # Images from test_dataloader are CURRENT_GAN_IMAGE_SIZE.
                # Resize them to SIMULATOR_IMAGE_SIZE (64x64) for the simulator.
                images_test = images_test.to(device)
                images_test_for_sim = resize_for_simulator(images_test) # Resize for simulator input
                targets_test = targets_test.to(device)
                outputs_test = simulator_model(images_test_for_sim)
                loss_test = criterion_S(outputs_test, targets_test)
                running_test_loss += loss_test.item() * images_test.size(0)
        # Ensure test_dataset is not empty before dividing
        if len(test_dataset) > 0:
            epoch_test_loss_val = running_test_loss / len(test_dataset)
            test_losses_sim.append(epoch_test_loss_val)
            print(f"Simulator Epoch [{epoch+1}/{NUM_EPOCHS_SIMULATOR}], Train Loss: {epoch_train_loss:.6f}, Test Loss: {epoch_test_loss_val:.6f}, Time: {time.time()-epoch_start_time:.2f}s")
        else:
            print(f"Simulator Epoch [{epoch+1}/{NUM_EPOCHS_SIMULATOR}], Train Loss: {epoch_train_loss:.6f}, Time: {time.time()-epoch_start_time:.2f}s (No test data for evaluation)")
    else:
        print(f"Simulator Epoch [{epoch+1}/{NUM_EPOCHS_SIMULATOR}], Train Loss: {epoch_train_loss:.6f}, Time: {time.time()-epoch_start_time:.2f}s (No test dataloader)")

print("--- Simulator Pre-training Finished. ---")

# Plotting Simulator Training and Test Loss
plt.figure(figsize=(10,5))
plt.title("Simulator Training & Test Loss")
plt.plot(train_losses_sim, label="Train MSE Loss")
if test_losses_sim:
    plt.plot(test_losses_sim, label="Test MSE Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR_SIM, "simulator_loss_curve_wgan_gp_conditional.png"))
plt.close() # Close figure to free memory

# Set simulator to evaluation mode and freeze its parameters
simulator_model.eval()
for param in simulator_model.parameters():
    param.requires_grad = False
print("Simulator model frozen for GAN training.")


# --- 6. Combined GAN Training Loop (WGAN-GP with Conditional G & C) ---
print("\n--- Starting Combined GAN Training (Conditional WGAN-GP with Simulator) ---")

# Lists to keep track of progress
G_losses = []
C_losses = [] # For Critic loss
Sim_losses_G_side = [] # Simulator loss when predicting on generator's output
iters = 0

print("Starting Training Loop...")
for epoch in range(NUM_EPOCHS_GAN):
    start_time_epoch = time.time() # Start timer for epoch
    for i, (real_images, target_absorbances) in enumerate(train_dataloader):
        current_batch_s = real_images.size(0)
        real_images = real_images.to(device) # Real images are CURRENT_GAN_IMAGE_SIZE
        target_absorbances = target_absorbances.to(device) # Ensure target_absorbances is on the GPU

        ############################
        # (1) Update Critic network: maximize C(x,y) - C(G(z,y),y) + GP
        # Paper's D_L loss
        ###########################
        for _ in range(N_CRITIC):
            netC.zero_grad()
            noise = torch.randn(current_batch_s, LATENT_DIM, 1, 1, device=device)
            # Generate fake images using noise AND target_absorbances as conditioning
            fake_images = netG(noise, target_absorbances).detach() # Detach to avoid training G on C's pass

            # Get scores for real and fake images (both conditioned on target_absorbances)
            real_scores = netC(real_images, target_absorbances)
            fake_scores_c = netC(fake_images, target_absorbances) # For Critic loss calculation

            # Gradient penalty
            gradient_penalty_val = compute_gradient_penalty(netC, real_images.data, fake_images.data, target_absorbances.data, current_batch_s)

            # Critic loss: WGAN loss + Gradient Penalty
            loss_c = fake_scores_c.mean() - real_scores.mean() + GP_WEIGHT * gradient_penalty_val
            loss_c.backward()
            optimizerC.step()

        # Step the Critic's LR scheduler after its updates for the batch
        scheduler_C.step()

        ############################
        # (2) Update Generator network: minimize -C(G(z,y),y) + λ * MSE(S(G(z,y)), y)
        # Paper's L_G = L_D + λ * L_S
        ###########################
        netG.zero_grad()
        # Generate a new batch of fake images for Generator update
        gen_noise = torch.randn(current_batch_s, LATENT_DIM, 1, 1, device=device)
        # Generate fake images using new noise AND target_absorbances as conditioning
        gen_fake_images = netG(gen_noise, target_absorbances) # These are CURRENT_GAN_IMAGE_SIZE images

        # 1. Critic Loss for Generator (L_D from paper, but WGAN style)
        fake_scores_g = netC(gen_fake_images, target_absorbances) # For G loss calculation, conditioned
        errG_C = -fake_scores_g.mean() # G wants to maximize Critic's output (minimize negative)

        # 2. Simulator Loss for Generator (L_S from paper)
        # IMPORTANT: Resize generated images to SIMULATOR_IMAGE_SIZE (64x64) before feeding to simulator
        gen_fake_images_for_sim = resize_for_simulator(gen_fake_images)
        simulator_output = simulator_model(gen_fake_images_for_sim) # S(G(z,y))
        errG_S = criterion_S(simulator_output, target_absorbances) # MSE loss with real targets (y)

        # Combined Generator Loss (L_G = L_C + λ L_S)
        errG = errG_C + LAMBDA_SIM_LOSS * errG_S
        errG.backward()
        optimizerG.step()

        # Step the Generator's LR scheduler after its update for the batch
        scheduler_G.step()

        # Output training stats
        # Logging includes LR for both G and C, and more detailed C(x) / C(G(z)) values
        if iters % 50 == 0 or (i == len(train_dataloader) - 1): # Log periodically and at the end of each batch
            print(f'[{epoch+1}/{NUM_EPOCHS_GAN}][{i}/{num_batches_per_epoch}] '
                  f'Loss_C: {loss_c.item():.4f} Loss_G: {errG.item():.4f} '
                  f'C(real): {real_scores.mean().item():.4f} C(fake): {fake_scores_c.mean().item():.4f} '
                  f'GP: {gradient_penalty_val.item():.4f} Sim_Loss_G_side: {errG_S.item():.4f} '
                  f'LR_G: {optimizerG.param_groups[0]["lr"]:.2e}, LR_C: {optimizerC.param_groups[0]["lr"]:.2e}')

        # Save Losses for plotting later
        G_losses.append(errG.item())
        C_losses.append(loss_c.item())
        Sim_losses_G_side.append(errG_S.item()) # This loss is for generated images

        iters += 1

    # Checkpointing and visualization after each epoch
    netG.eval() # Set G to eval mode for consistent batch norm behavior during inference
    with torch.no_grad():
        # Generate images using fixed noise and fixed absorbance target
        fake_fixed_noise_images = netG(fixed_noise, fixed_absorbance_for_gen).detach().cpu()
        vutils.save_image(fake_fixed_noise_images,
                          f'{OUTPUT_DIR_GAN}/fake_samples_epoch_{epoch+1:03d}.png',
                          normalize=True)
    netG.train() # Set G back to train mode

    # Save model checkpoints
    if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS_GAN:
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': netG.state_dict(),
            'critic_state_dict': netC.state_dict(),
            'simulator_state_dict': simulator_model.state_dict(),
            'optimizer_G_state_dict': optimizerG.state_dict(),
            'optimizer_C_state_dict': optimizerC.state_dict(),
            'optimizer_sim_state_dict': optimizer_sim.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_C_state_dict': scheduler_C.state_dict(),
            'G_losses': G_losses,
            'C_losses': C_losses,
            'Sim_losses_G_side': Sim_losses_G_side,
            'train_losses_sim': train_losses_sim,
            'test_losses_sim': test_losses_sim,
            'fixed_noise': fixed_noise,
            'fixed_absorbance_for_gen': fixed_absorbance_for_gen,
            'iters': iters,
        }, os.path.join(OUTPUT_DIR_GAN, "checkpoints", f'wgan_gp_combined_ckpt_epoch_{epoch+1}.pth'))
        print(f"Saved models and training state at epoch {epoch+1}")

    epoch_duration = time.time() - start_time_epoch
    print(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")


print("--- Combined WGAN-GP Training Finished. ---")

# --- Plotting Results ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Generator and Critic Loss During Conditional WGAN-GP Training")
plt.plot(G_losses, label="G Loss (Combined)")
plt.plot(C_losses, label="Critic Loss (WGAN-GP)")
plt.plot(Sim_losses_G_side, label="Simulator Loss on G Images (for G)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
# Re-create logged LR for plotting, as it's not explicitly stored iteration-wise for every single step
# The scheduler.step() is called every iteration.
logged_lr_g = []
logged_lr_c = []
# Create dummy optimizers/schedulers to simulate the LR progression for plotting
dummy_optimizer_G = optim.Adam(netG.parameters(), lr=INITIAL_LR_G, betas=(BETA1_ADAM, BETA2_ADAM))
dummy_scheduler_G = CosineAnnealingLR(dummy_optimizer_G, T_max=DECAY_STEPS_TOTAL, eta_min=INITIAL_LR_G*0.001)
dummy_optimizer_C = optim.Adam(netC.parameters(), lr=INITIAL_LR_C, betas=(BETA1_ADAM, BETA2_ADAM))
dummy_scheduler_C = CosineAnnealingLR(dummy_optimizer_C, T_max=DECAY_STEPS_TOTAL, eta_min=INITIAL_LR_C*0.001)

for _ in range(iters): # Simulate all iterations that occurred
    logged_lr_g.append(dummy_scheduler_G.get_last_lr()[0])
    logged_lr_c.append(dummy_scheduler_C.get_last_lr()[0])
    dummy_scheduler_G.step()
    dummy_scheduler_C.step()

plt.plot(np.arange(len(logged_lr_g)), logged_lr_g, label="Scheduled LR (G)")
plt.plot(np.arange(len(logged_lr_c)), logged_lr_c, label="Scheduled LR (C)", linestyle='--')
plt.title("Learning Rate Schedule (Cosine Annealing)")
plt.xlabel("Iterations")
plt.ylabel("Learning Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR_GAN, "gan_loss_and_lr_curve_wgan_gp_conditional.png"))
plt.close() # Close figure to free memory


# --- Final Evaluation on Generator Quality and Simulator Performance on Generated Images ---

print("\n--- Final Evaluation of Generator and Simulator on Generated Images ---")
netG.eval()
simulator_model.eval() # Ensure simulator is still in eval mode

# Generate a larger batch of fake images for final evaluation
num_eval_images = 100
# For evaluation, we need target absorbances to condition the generator.
# Let's take the first 'num_eval_images' absorbances from the training dataset.
eval_absorbances = []
# Ensure we take enough samples without going out of bounds
num_samples_to_take = min(num_eval_images, len(train_dataset))
for i in range(num_samples_to_take):
    eval_absorbances.append(train_dataset[i][1])
# If num_eval_images > len(train_dataset), repeat the last available absorbance
if num_samples_to_take < num_eval_images:
    last_absorbance = train_dataset[num_samples_to_take - 1][1] if num_samples_to_take > 0 else torch.zeros(NUM_ANGLES)
    for _ in range(num_eval_images - num_samples_to_take):
        eval_absorbances.append(last_absorbance)

eval_absorbances = torch.stack(eval_absorbances).to(device)
eval_noise = torch.randn(num_eval_images, LATENT_DIM, 1, 1, device=device)

with torch.no_grad():
    generated_images = netG(eval_noise, eval_absorbances).detach() # CURRENT_GAN_IMAGE_SIZE images
    # Resize generated images for simulator before prediction
    generated_images_for_sim = resize_for_simulator(generated_images) # Resized to SIMULATOR_IMAGE_SIZE
    predicted_absorbances = simulator_model(generated_images_for_sim).cpu().numpy()

# Save some generated images
vutils.save_image(generated_images[:min(64, num_eval_images)],
                   f'{OUTPUT_DIR_GAN}/final_generated_samples_wgan_gp_conditional.png',
                   normalize=True)

# Analyze predicted absorbances from generated images
print(f"\nSample Predicted Absorbances from Generated Images (first {min(5, num_eval_images)}):\n")
for i in range(min(5, num_eval_images)):
    print(f"Sample {i+1}: {[f'{x:.3f}' for x in predicted_absorbances[i]]}")

print(f"\nAll outputs saved in: {OUTPUT_DIR_BASE}")
