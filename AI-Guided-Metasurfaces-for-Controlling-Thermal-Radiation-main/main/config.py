# config.py
import torch
import os
import matplotlib

matplotlib.use('Agg')

# --- 0. Core Configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAINING_MODE = 'CONSTANT_TARGET' # Options: 'CONDITIONAL', 'CONSTANT_TARGET'.

# --- 1. Data Paths---
# Using absolute paths for your Linux environment to ensure files are always found.
PROJECT_ROOT = "C:/Users/travi/Projects/AI-Guided-Metasurfaces-for-Controlling-Thermal-Radiation"
# >>> CORRECTED: Removed the duplicated "Generated_Images" from the path.
IMAGE_FOLDER_PATH = os.path.join(PROJECT_ROOT, "Data/Data_Generated_Images")
METADATA_FILE = os.path.join(PROJECT_ROOT, "metasurface_absorbance_compiled_final.csv")
TARGET_RESPONSES_FILE = os.path.join(PROJECT_ROOT, "target_responses.csv")

#If path is provided, simulator training will be skipped. Set to None to run the training loop.
PRETRAINED_SIMULATOR_PATH = "/home/travis/Projects/Metasurfaces/AI-Guided-Metasurfaces-for-Controlling-Thermal-Radiation/best_simulator_model.pth"

# --- 2. Image and Data Dimensions ---
CURRENT_GAN_IMAGE_SIZE = 64
SIMULATOR_IMAGE_SIZE = 64
CHANNELS = 1
NUM_ANGLES = 15
TEST_SPLIT_SIZE = 0.10

# --- 3. Model Parameters ---
LATENT_DIM = 128
GF = 64
DF = 64
SIM_NDF = 256
DROPOUT_RATE_SIM = 0.3 #0.3 is best

# --- 4. Training Parameters ---
BATCH_SIZE = 128
WORKERS = 2
NUM_EPOCHS_SIMULATOR = 500 if TRAINING_MODE == 'CONSTANT_TARGET' else 300
NUM_EPOCHS_GAN = 1 if TRAINING_MODE == 'CONSTANT_TARGET' else 15000

# --- SIMULATOR LOSS & OPTIMIZER CONFIG ---
LAMBDA_GRAD = 1 #Previouslly 1.
WEIGHT_DECAY_SIMULATOR = 1e-4 #Previously 1e-5, this is better.

# --- 5. GAN Optimizer and Loss Parameters ---
LEARNING_RATE_SIM = 0.0002
INITIAL_LR_G = 0.0001
INITIAL_LR_C = 0.00005
BETA1_ADAM = 0.0
BETA2_ADAM = 0.9
N_CRITIC = 5
GP_WEIGHT = 10.0
LAMBDA_SIM_LOSS = 0.000025

# --- 6. Output Directory ---
OUTPUT_DIR_BASE = "output_constant_target" if TRAINING_MODE == 'CONSTANT_TARGET' else "output_conditional"

# --- 7. Evaluation Parameters ---
NUM_IMAGES_PER_ABSORPTION_TEST = 5
NUM_EVAL_IMAGES_FOR_CONSTANT_TARGET = 100
