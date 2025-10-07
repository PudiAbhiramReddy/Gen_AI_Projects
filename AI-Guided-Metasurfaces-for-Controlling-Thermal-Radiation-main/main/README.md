# AI-Guided Metasurfaces for Controlling Thermal Radiation

This repository contains the core codebase for an AI-guided pipeline designed to discover and design metasurface structures for controlling thermal radiation. The project leverages deep learning models, including Generative Adversarial Networks (GANs) and a CNN-based simulator, to explore the vast design space of metasurfaces and predict their optical properties.

## Project Structure and File Descriptions

This section provides an in-depth look into the key Python files within this project, detailing their functionalities, architectural components, and technical aspects.

---

### `main/config.py`

This file serves as the central configuration hub for the entire project. It defines global parameters, hyperparameters, and file paths that are used across different modules, ensuring consistency and ease of modification.

#### Technical Details:
* **Data Paths**: Specifies the root directory for generated images (`IMAGE_FOLDER_PATH`), and the paths for training and testing metadata CSV files (`TRAIN_METADATA_FILE`, `TEST_METADATA_FILE`). These paths are crucial for data loading.
* **Model Parameters**:
    * `CHANNELS`: Number of input channels for images (e.g., `1` for grayscale).
    * `NUM_ANGLES`: The dimension of the absorbance spectrum output (number of data points in the spectrum).
    * `LATENT_DIM`: Dimension of the latent space vector used by the Generator (noise input).
    * `GF` (Generator Feature multiplier): Base feature size for the Generator network, determining its width.
    * `DF` (Discriminator/Critic Feature multiplier): Base feature size for the Critics, determining their width.
    * `SIM_NDF` (Simulator Network Depth/Feature multiplier): Base feature size for the `RobustSimulatorCNN`, determining its width. This parameter has been modified to increase the simulator's capacity.
* **Image Dimensions**:
    * `SIMULATOR_IMAGE_SIZE`: The target size (e.g., `64x64`) to which images are resized before being fed into the simulator.
    * `CURRENT_GAN_IMAGE_SIZE`: The target size for GAN-generated images.
* **Training Parameters**:
    * `BATCH_SIZE`: Number of samples processed in one forward/backward pass.
    * `NUM_WORKERS`: Number of subprocesses to use for data loading, improving loading speed.
    * `NUM_EPOCHS`: Total number of training iterations over the dataset.
    * `LEARNING_RATE`: Step size for optimizer updates.
    * `BETA1`, `BETA2`: Parameters for the Adam optimizer.
    * `TRAINING_MODE`: A critical switch determining the pipeline's behavior, either `'CONSTANT_TARGET'` (for unconditional GAN and simulator training) or `'CONDITIONAL'` (for conditional GAN training).
* **Output Paths**: Defines where trained models, generated images, and plots will be saved.

---

### `main/data_loader.py`

This file is responsible for loading and preprocessing the metasurface image data and their corresponding absorbance spectra. It defines custom PyTorch `Dataset` classes and a function to set up `DataLoader` instances, now supporting images organized in subfolders.

#### Architectural Details:
* **`_find_image_files_recursive(folder_path)` Function (Helper)**:
    * **Description**: A utility function that recursively walks through a given `folder_path` and its subdirectories to find all files ending with common image extensions (e.g., `.png`, `.jpg`).
    * **Technical Details**: Uses `os.walk` to traverse the directory tree and `os.path.basename` to extract just the filename, storing the full path. This enables robust loading of images regardless of their subfolder location.
* **`ImageOnlyDataset(Dataset)` Class**:
    * **Purpose**: Designed to load images for the unconditional Critic training in `'CONSTANT_TARGET'` mode. It retrieves images without requiring corresponding labels.
    * **Technical Details**:
        * `__init__(self, image_folder_path, transform=None)`: Initializes by calling `_find_image_files_recursive` to populate `self.image_files` with full paths to all relevant images.
        * `__len__()`: Returns the total number of image files found.
        * `__getitem__(self, idx)`: Loads a single image by its full path, converts it to grayscale (`'L'`), and applies specified transformations.
* **`SingleFolderMetasurfaceDataset(Dataset)` Class**:
    * **Purpose**: Loads metasurface images and their associated absorbance spectra from a metadata CSV file and an image folder. Used for simulator training and conditional GAN training.
    * **Technical Details**:
        * `__init__(self, metadata_file, image_folder_path, transform=None, num_angles=NUM_ANGLES)`:
            * Reads the `metadata_file` (CSV), which is expected to contain an 'Image Name' column and columns for absorbance values.
            * **`self.image_path_map`**: A crucial dictionary mapping *base image filenames* (e.g., `circle_0000.png`) to their *full paths* found recursively within `image_folder_path` using `_find_image_files_recursive`. This ensures correct image lookup even if they reside in various subfolders.
            * **Data Filtering**: The `metadata_df` is filtered to include only entries for which a corresponding image file was successfully located in `self.image_path_map`. Warnings are printed if entries are discarded due to missing images.
        * `__len__()`: Returns the number of entries in the filtered metadata DataFrame.
        * `__getitem__(self, idx)`:
            * Retrieves the image filename from the CSV.
            * Uses `self.image_path_map` to get the actual full path to the image file, robustly handling subfolders.
            * Loads the image, converts to grayscale, and applies defined transformations.
            * Extracts the absorbance vector (NumPy array) from the CSV row and converts it to a PyTorch tensor.
            * Returns the image and the absorbance tensor.
* **`get_dataloaders(...)` Function**:
    * **Purpose**: Orchestrates the creation of appropriate `Dataset` and `DataLoader` instances based on the `TRAINING_MODE` defined in `config.py`.
    * **Technical Details**:
        * Defines separate `torchvision.transforms.Compose` pipelines for GAN and simulator images, including resizing, cropping, conversion to tensor, and normalization.
        * Based on the `mode`, it instantiates the correct datasets (`ImageOnlyDataset` or `SingleFolderMetasurfaceDataset`) and wraps them in `DataLoader` for efficient batching, shuffling, and multi-threaded loading (`num_workers`).
        * Includes robust error handling for `FileNotFoundError` or `ValueError` (e.g., empty datasets).
* **`resize_for_simulator(images)` Function**: A simple helper function for resizing images specifically to the simulator's required input size.

---

### `main/models.py`

This file defines the neural network architectures for the Generator, two Critic (Discriminator) models, and the `RobustSimulatorCNN`, which is central to predicting metasurface properties.

#### Architectural Details:
* **`SEBlock(nn.Module)` Class (NEW)**:
    * **Purpose**: Implements a Squeeze-and-Excitation (SE) Block, an attention mechanism that allows the network to perform dynamic channel-wise feature recalibration.
    * **Technical Details**:
        * `avg_pool = nn.AdaptiveAvgPool2d(1)`: The "Squeeze" operation, performing global average pooling to get channel-wise statistics.
        * `fc = nn.Sequential(...)`: The "Excitation" operation, consisting of two linear layers with ReLU and Sigmoid activations. These layers learn non-linear interactions between channel feature maps and output channel-wise attention weights.
        * `forward(self, x)`: The input features `x` are scaled by the learned attention weights `y`, amplifying important features and suppressing less useful ones.
* **`Generator(nn.Module)` Class**:
    * **Purpose**: Generates metasurface images from a random noise vector and a conditional absorbance spectrum. Used in GAN training.
    * **Architecture**: A Deep Convolutional GAN (DCGAN) style generator utilizing `nn.ConvTranspose2d` (transposed convolutions or "deconvolutions") for progressive upsampling.
    * **Technical Details**:
        * **Input**: Takes a `noise` (latent vector) and an `absorbance_vector`. The absorbance vector is spatially expanded and concatenated with the noise, providing the conditional information for image generation.
        * **Layers**: Begins with a `ConvTranspose2d` that directly upsamples a 1x1 input. Subsequent layers use `ConvTranspose2d` with a stride of 2 for further spatial upsampling.
        * **Activations**: `nn.ReLU` is used for hidden layers. The final output layer employs `nn.Tanh` to produce pixel values normalized in the range `[-1, 1]`.
        * **Normalization**: `nn.BatchNorm2d` is applied after most transposed convolutional layers to stabilize training and improve convergence.
* **`ConditionalCritic(nn.Module)` Class**:
    * **Purpose**: Discriminates between real and fake images, conditioned on an absorbance vector. This critic is used specifically in the `'CONDITIONAL'` GAN pipeline.
    * **Architecture**: A deep convolutional discriminator employing `nn.Conv2d` layers for downsampling and feature extraction.
    * **Technical Details**:
        * **Input**: Receives both an `image_input` and an `absorbance_vector`. The absorbance vector is expanded spatially to match the image dimensions and then concatenated channel-wise with the image input, allowing the critic to evaluate the image based on the given condition.
        * **Layers**: Consists of a series of `nn.Conv2d` layers with a stride of 2 for progressive downsampling.
        * **Activations**: `nn.LeakyReLU(0.2)` is used throughout the network as the activation function, which helps in preventing vanishing gradient issues.
        * **Output**: Produces a single scalar value representing the discriminator's "realness" score for the input image-condition pair (or a Wasserstein distance score in WGAN-GP).
* **`UnconditionalCritic(nn.Module)` Class**:
    * **Purpose**: Discriminates between real and fake images *without* any conditional input. This critic is used in the `'CONSTANT_TARGET'` GAN pipeline.
    * **Architecture**: Functionally similar to the `ConditionalCritic` but designed to accept only an image input.
    * **Technical Details**: Follows a series of `nn.Conv2d` layers with `nn.LeakyReLU` activations. Outputs a single scalar score indicating the perceived realness of the input image.
* **`RobustSimulatorCNN(nn.Module)` Class (IMPROVED)**:
    * **Purpose**: The core physics-based simulator, designed to predict the absorbance spectrum (`NUM_ANGLES` output values) of a given metasurface image.
    * **Architecture**: Comprises a powerful convolutional neural network (`cnn_layers`) for robust feature extraction from the image, followed by fully connected layers (`fc_layers`) to map these features to the absorbance spectrum.
    * **Technical Details**:
        * **`cnn_layers`**:
            * **Convolutional Blocks**: A sequence of `nn.Conv2d` layers performs hierarchical feature learning and spatial downsampling. Each block typically includes `nn.BatchNorm2d` (except the first layer as designed) and `nn.LeakyReLU(0.2)` activation.
            * **Kernel and Stride**: Uses `kernel_size=4` and `stride=2` for efficient spatial reduction, with the final convolutional layer (`stride=1, padding=0`) aggregating features into a 1x1 spatial dimension.
            * **Squeeze-and-Excitation (SE) Blocks**: **Crucially, an `SEBlock` is inserted after each `LeakyReLU` activation within the `cnn_layers`.** These attention modules allow the network to dynamically learn which feature channels are most relevant at each stage of processing, enhancing its ability to focus on critical image characteristics that drive specific absorbance profiles, particularly high-magnitude peaks. This directly addresses the goal of improving peak prediction accuracy.
            * **`SIM_NDF` (Increased Width)**: The `ndf` parameter (sourced from `SIM_NDF` in `config.py`) has been increased, effectively making the convolutional layers wider and allowing them to extract a richer set of features.
        * **Feature Flattening**: The output of the `cnn_layers` (a multi-channel 1x1 feature map) is flattened into a 1D vector.
        * **`fc_layers`**:
            * **Linear Mapping**: Consists of multiple `nn.Linear` layers that map the high-dimensional flattened CNN features to the `num_outputs` (absorbance values at `NUM_ANGLES` wavelengths).
            * **Activations**: `nn.ReLU(True)` is used for non-linearity in the hidden FC layers.
            * **Regularization**: `nn.Dropout(0.5)` is applied after the first two FC layers to regularize the model and reduce overfitting by randomly setting a fraction of neurons to zero during training.
            * **Output Activation**: The final layer uses `nn.Sigmoid()` activation to ensure the predicted absorbance values are constrained within the valid range of 0 to 1.
        * **Dynamic CNN Output Size Calculation**: A `torch.no_grad()` block dynamically calculates the `cnn_out_size` by passing a dummy input through the `cnn_layers`. This makes the architecture robust to changes in image size or convolutional layer configurations, as the input dimension for the first `nn.Linear` layer is automatically determined.

---

### `main/training_loops.py`

This file encapsulates the core training logic for both the simulator and the GAN models. It orchestrates the forward passes, calculates losses, performs backpropagation, and manages optimizer steps.

#### Technical Details:
* **`train_simulator(...)` Function**:
    * **Purpose**: Manages the training loop for the `RobustSimulatorCNN`.
    * **Process**: Iterates through epochs and batches from the training DataLoader. For each batch, it feeds images to the `RobustSimulatorCNN` to get predicted absorbance, calculates the loss (typically `nn.MSELoss`) between predictions and ground truth, performs backpropagation (`loss.backward()`), and updates simulator weights (`optimizer_sim.step()`). It also periodically calls `evaluate_simulator` to monitor performance.
* **`train_gan_constant_target(...)` Function**:
    * **Purpose**: Manages the training loop for the unconditional GAN in the `'CONSTANT_TARGET'` pipeline.
    * **Process**: Involves alternating training steps for both the Generator and the `UnconditionalCritic`.
        * **Critic Training**: The Critic is trained to distinguish between real images from the `gan_unconditional_loader` and fake images generated by the `Generator`. Critic loss (e.g., Wasserstein loss with gradient penalty) is calculated, and Critic weights are updated.
        * **Generator Training**: The Generator is trained to produce images that are convincing enough to fool the `UnconditionalCritic`. Generator loss is calculated based on the Critic's output for fake images, and Generator weights are updated.
* **`train_gan_conditional(...)` Function**:
    * **Purpose**: Manages the training loop for the conditional GAN in the `'CONDITIONAL'` pipeline.
    * **Process**: Similar to `train_gan_constant_target`, but both the `Generator` and `ConditionalCritic` receive absorbance vectors as conditioning input. The Generator takes noise and absorbance to generate images. The Critic takes both real/fake images and their corresponding absorbance to discriminate their realness and condition adherence.
* **`evaluate_simulator(...)` Function**:
    * **Purpose**: Evaluates the `RobustSimulatorCNN`'s performance on the unseen test dataset.
    * **Process**: Calculates test loss (e.g., RMSE, Accuracy) by comparing predictions to ground truth absorbance values on test data. `torch.no_grad()` is used to disable gradient calculations during evaluation, optimizing performance.

---

### `main/losses_optimizers.py`

This file centralizes the definition of loss functions and optimizers used throughout the training processes for all models.

#### Technical Details:
* **Loss Functions**:
    * `nn.MSELoss()`: Mean Squared Error, serving as the primary loss function for the simulator's regression task, measuring the average squared difference between predicted and true absorbance.
    * Other loss components are implicitly used within the GAN training loops (e.g., components for Wasserstein loss, gradient penalty) though not explicitly defined here as separate `nn.Module` classes.
* **Optimizers**:
    * `torch.optim.Adam()`: The Adam optimizer is consistently employed for all model components (simulator, Generator, Critics) due to its efficiency and adaptive learning rate capabilities. It is configured with parameters (`LEARNING_RATE`, `BETA1`, `BETA2`) loaded from `config.py`.

---

### `main/utils.py`

This file contains various utility functions that provide supporting functionalities across the main project pipeline, including device management, weight initialization, and model checkpointing.

#### Technical Details:
* **`get_device()`**: Automatically detects and returns the optimal computation device (`'cuda'` for GPU if available, otherwise `'cpu'`), enabling flexible execution across different hardware configurations.
* **`weights_init(m)`**: A custom weight initialization function. It applies specific initialization strategies (e.g., normal distribution for convolutional/linear layers and constant for BatchNorm layers) to the network modules, crucial for stable and effective training, particularly in GANs.
* **`save_checkpoint(state, filename)`**: Facilitates saving the current state of models and their optimizers, along with epoch information, allowing for training to be paused and resumed later.
* **`load_checkpoint(checkpoint, model, optimizer, device)`**: Loads a previously saved model checkpoint onto the specified device, restoring the model and optimizer states for continued training or inference.
* Other helper functions may be included for tasks such as data logging, plotting, or general tensor manipulations.

---

### `main/graph_output.py`

This file is dedicated to the visualization and saving of training results and model outputs, providing insights into performance and generated data.

#### Technical Details:
* Contains functions for:
    * **Spectral Plotting**: Generating plots that compare predicted absorbance spectra against ground truth values for test samples.
    * **Image Saving**: Saving generated metasurface images during GAN training to monitor visual quality.
    * **Performance Visualization**: Creating composite plots that summarize simulator performance on specific test samples, including visual comparisons, accuracy, and RMSE metrics.
    * **Metric Logging**: Recording and potentially visualizing training and test losses, accuracy, and other metrics over training epochs.

---

### `main/main.py`

This file serves as the primary entry point for the entire project. It orchestrates the end-to-end metasurface generation pipeline, from initial setup to model training and output management.

#### Technical Details:
* **Argument Parsing**: Typically includes logic to parse command-line arguments, allowing flexible configuration of training parameters and modes without direct code modification.
* **Device Initialization**: Initializes the computation device (CPU or GPU) using `get_device()` from `utils.py`.
* **Output Directory Setup**: Creates necessary directories for saving models, plots, and generated data.
* **Data Loading**: Calls `get_dataloaders()` from `data_loader.py` to prepare the training and testing datasets.
* **Model Initialization**: Instantiates the `Generator`, `Critic` (either `ConditionalCritic` or `UnconditionalCritic` based on `TRAINING_MODE`), and `RobustSimulatorCNN` models, and applies `weights_init()` for proper initialization.
* **Optimizer and Loss Setup**: Configures the optimizers and relevant loss functions from `losses_optimizers.py` for each model.
* **Main Training Loop**: Contains the central loop that iterates over `NUM_EPOCHS`. Within each epoch, it calls the appropriate training functions (e.g., `train_simulator`, `train_gan_constant_target`, or `train_gan_conditional`) from `training_loops.py`.
* **Checkpointing and Logging**: Manages saving model checkpoints periodically and logging training progress (losses, metrics, time per epoch) to console or log files.

---

### `python/data_generation/*.py` (e.g., `shapeGenerator.py`, `blobCreator.py`, `arrays.py`)

These scripts are foundational for generating the synthetic metasurface images that serve as input for your deep learning models. They define the algorithms for creating diverse geometric shapes.

#### Technical Details:
* **Image Creation Libraries**: Utilize libraries such as `Pillow (PIL)` or `OpenCV` for image manipulation and rendering.
* **Geometric Primitives**: Implement logic to draw and fill basic geometric shapes like circles, squares, rectangles, and more complex patterns for "blobs" or "arrays."
* **Parameter Randomization/Systematization**: Parameters such as size, position, rotation, and fill properties are either randomized within defined ranges or varied systematically to ensure a rich and diverse dataset that covers a wide range of design possibilities.
* **Data Association**: These scripts are often responsible for generating the `.png` image files and, crucially, for creating or updating metadata files (like your `metasurface_absorbance_train.csv`) that link each generated image's filename to its corresponding simulated physical properties (e.g., absorbance spectrum), which are obtained from physics simulations.