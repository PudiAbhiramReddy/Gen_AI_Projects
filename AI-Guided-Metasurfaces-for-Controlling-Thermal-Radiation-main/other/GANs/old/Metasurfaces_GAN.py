import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode # For better readability
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image # Not strictly needed for ImageFolder but good to have for image ops

from google.colab import drive
drive.mount('/content/drive')


import zipfile
import os

# Path to zip file
zip_path = '/content/drive/MyDrive/Data (1).zip'

# Destination folder to unzip
extract_to = '/content/drive/My Drive/MetaSurfaces_Final1/'

# Unzipping
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Unzipped successfully to:", extract_to)


# --- 0. Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Hyperparameters ---
# IMPORTANT: Set this to the PARENT folder containing your category subfolders
dataroot = "/content/drive/MyDrive/MetaSurfaces_Final1/Data" # e.g., "/content/My_Metasurface_Data"

workers = 2             # Number of workers for dataloader
batch_size = 64         # Batch size during training (can reduce if OOM on Colab free tier)
image_size = 64         # Spatial size of training images AFTER downscaling.
original_image_size = 256 # Just for reference if needed
nc = 1                  # Number of channels in the training images (1 for grayscale).
nz = 100                # Size of z latent vector (i.e. size of generator input)
ngf = 64                # Size of feature maps in generator
ndf = 64                # Size of feature maps in discriminator
num_epochs = 200        # Number of training epochs (start with this, may need many more)
lr_d = 0.00005 # Was 0.0002
lr_g = 0.0002  # Keep G's LR or slightly increase
beta1 = 0.5             # Beta1 hyperparameter for Adam optimizers
ngpu = 1                # Number of GPUs available.



# Create output directories
output_dir = "gan_output_metasurfaces"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)


# --- 2. Data Loading and Preprocessing ---
# Assumes dataroot points to a folder, and inside it are subfolders for each category
# e.g., dataroot/Category_A/image1.png, dataroot/Category_B/image2.png
try:
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                                   transforms.CenterCrop(image_size), # Ensures exactly image_size x image_size
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)), # Normalize to [-1, 1]
                               ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers, pin_memory=True) # pin_memory for faster transfer to GPU

    # Plot some training images to verify
    if len(dataset) > 0:
        print(f"Dataset loaded successfully with {len(dataset)} images.")
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(f"Training Images (Downscaled to {image_size}x{image_size})")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig(os.path.join(output_dir, "sample_training_images.png"))
        plt.show()
    else:
        print("Dataset is empty. Please check the 'dataroot' path and folder structure.")
        print(f"Current dataroot: {dataroot}")
        print("Expected structure: dataroot/category_folder/image.png")
        exit()

except FileNotFoundError:
    print(f"Error: dataroot '{dataroot}' not found. Please set it correctly.")
    print("Expected structure: dataroot/category_folder/image.png")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    print(f"Current dataroot: {dataroot}")
    print("Please check your 'dataroot' path and image integrity.")
    exit()

# --- 3. Model Definitions ---

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu_g): # Renamed ngpu to ngpu_g to avoid conflict
        super(Generator, self).__init__()
        self.ngpu = ngpu_g
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input_val): # Renamed input to input_val
        return self.main(input_val)

# Create the generator
netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
# print(netG) # Uncomment to see the generator architecture



# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu_d): # Renamed ngpu to ngpu_d
        super(Discriminator, self).__init__()
        self.ngpu = ngpu_d
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_val): # Renamed input to input_val
        return self.main(input_val)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
# print(netD) # Uncomment to see the discriminator architecture

# --- 4. Loss Functions and Optimizers ---
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize progression
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

# --- 5. Training Loop ---
img_list = []
G_losses = []
D_losses = []
iters = 0
# Change label definitions
real_label_val = 0.9  # Instead of 1.0
fake_label_val = 0.1  # Instead of 0.0 (for D training on fakes)
# Note: When G is trained, it still tries to make D output "real" (target 1.0 or 0.9)

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # In the training loop:
        # When training D with real:
        label = torch.full((b_size,), real_label_val, dtype=torch.float, device=device)
        #label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Construct fake batch with G
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        #label.fill_(fake_label)
        # When training D with fake:
        label.fill_(fake_label_val) # Use fake_label_val here
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1) # .detach() so G is not updated
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # # (2) Update G network: maximize log(D(G(z)))
        # netG.zero_grad()
        # #label.fill_(real_label)  # fake labels are real for generator cost
        # # When training G:
        # label.fill_(real_label_val) # G still aims for D to say "real"
        # # Since we just updated D, perform another forward pass of all-fake batch through D
        # output = netD(fake).view(-1)
        # errG = criterion(output, label)
        # errG.backward()
        # D_G_z2 = output.mean().item()
        # # Update G
        # optimizerG.step()

        # Inside the main epoch loop, after D's update:
        k_steps_g = 2 # Try 2 or 3
        for g_step in range(k_steps_g):
              netG.zero_grad()
              # It's often good to generate new noise and fakes for each G update
              # to get more diverse gradients, though using the same 'fake' from
              # D's step is also a common starting point.
              noise_g_step = torch.randn(b_size, nz, 1, 1, device=device)
              fake_g_step = netG(noise_g_step)
              label.fill_(real_label_val) # Use the target "real" label
              output = netD(fake_g_step).view(-1)
              errG = criterion(output, label)
              errG.backward()
              # D_G_z2 = output.mean().item() # This D_G_z2 will be for the last G step
              optimizerG.step()
              # Only update D_G_z2 on the last G step so it reflects the final state for this batch
              if g_step == k_steps_g - 1:
                    D_G_z2 = output.mean().item() # D's output for fakes G was just trained on

        # Output training stats
        if i % 50 == 0: # Print every 50 batches
            print(f'[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 200 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)): # Save every 200 iterations or at the very end
            with torch.no_grad():
                fake_display = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_display, padding=2, normalize=True))
            vutils.save_image(fake_display,
                    f"{output_dir}/images/fake_samples_epoch_{epoch+1}_iter_{iters}.png",
                    padding=2, normalize=True)
        iters += 1

    # Save model checkpoints after each epoch (or less frequently if needed)
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1: # Save every 10 epochs or at the end
        torch.save(netG.state_dict(), f'{output_dir}/checkpoints/netG_epoch_{epoch+1}.pth')
        torch.save(netD.state_dict(), f'{output_dir}/checkpoints/netD_epoch_{epoch+1}.pth')
        print(f"Saved checkpoints for epoch {epoch+1}")

print("Training Finished.")

# --- 6. Plotting Results ---
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{output_dir}/loss_curve.png")
plt.show()



# Grab a batch of real images from the dataloader
real_batch_display = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,8)) # Adjusted figure size for two subplots
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images (Sample)")
plt.imshow(np.transpose(vutils.make_grid(real_batch_display[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last set of fixed_noise
if img_list: # Check if img_list is not empty
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images (Last Iteration)")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
else:
    print("No fake images were generated to display (img_list is empty).")

plt.tight_layout() # Adjust layout to prevent overlap
plt.savefig(f"{output_dir}/real_vs_fake_comparison.png")
plt.show()

print(f"Outputs saved in directory: {output_dir}")
