# This file experiments with different loss functions, but overall performs worse than gan.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import InterpolationMode
from multiprocessing import freeze_support

def main():
    # --- 0. Setup Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 1. Hyperparameters ---
    dataroot = "/Users/travis/Desktop/Projects/AI Summer 25/Code/GAN/Data"

    workers = 2
    batch_size = 64
    image_size = 256           # <-- Updated for 256x256 output
    nc = 1
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 200
    lr_d = 0.00005
    lr_g = 0.0002
    beta1 = 0.5
    ngpu = 1

    output_dir = "gan_output_metasurfaces_256"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # --- 2. Data Loading and Preprocessing ---
    dataset = dset.ImageFolder(root=dataroot,
                              transform=transforms.Compose([
                                  transforms.Grayscale(num_output_channels=1),
                                  transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                                  transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                              ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers, pin_memory=True)

    if len(dataset) == 0:
        print("Dataset is empty. Check dataroot and folder structure.")
        exit()

    # --- 3. Model Definitions ---

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False), # 1x1 -> 4x4
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False), # 4x4 -> 8x8
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), # 8x8 -> 16x16
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # 16x16 -> 32x32
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), # 32x32 -> 64x64
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False), # 64x64 -> 128x128
                nn.BatchNorm2d(ngf // 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False), # 128x128 -> 256x256
                nn.Tanh()
            )

        def forward(self, input_val):
            return self.main(input_val)

    netG = Generator(ngpu).to(device)
    if (device.type == 'cuda' or device.type == 'mps') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input 256x256
                nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),  # 256 -> 128
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),  # 128 -> 64
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 64 -> 32
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 32 -> 16
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 16 -> 8
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),  # 8 -> 4
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),  # 4 -> 1
                nn.Sigmoid()
            )

        def forward(self, input_val):
            return self.main(input_val)

    netD = Discriminator(ngpu).to(device)
    if (device.type == 'cuda' or device.type == 'mps') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)

    # --- 4. Loss and Optimizers ---
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label_val = 0.9
    fake_label_val = 0.1

    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

    # --- 5. Training Loop ---
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label_val, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label_val)

            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label_val)

            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if i % 50 == 0:
                print(f"[{epoch + 1}/{num_epochs}][{i}/{len(dataloader)}] "
                      f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                      f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_grid = vutils.make_grid(fake, padding=2, normalize=True)
                vutils.save_image(img_grid, os.path.join(output_dir, "images", f"epoch_{epoch}_iter_{iters}.png"))
                img_list.append(img_grid)

            iters += 1

        torch.save(netG.state_dict(), os.path.join(output_dir, "checkpoints", f"netG_epoch_{epoch}.pth"))
        torch.save(netD.state_dict(), os.path.join(output_dir, "checkpoints", f"netD_epoch_{epoch}.pth"))

    print("Training finished.")

if __name__ == '__main__':
    freeze_support()
    main()
