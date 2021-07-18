#
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
# import torchvision.transforms as transforms

# Losses
from torch.nn.modules.loss import BCEWithLogitsLoss, L1Loss

# Private Modules
from Discriminator import Discriminator
from Generator import Generator
# from train_config import transforms
from Data import CityScapesDataset
import train_config as config


def train_loop(discriminator: nn.Module, generator: nn.Module, loader: DataLoader, opt_disc: optim, opt_gen: optim, l1_loss, bce, g_scaler, d_scaler):

    for index, (x, y) in enumerate(loader):
        print(f"step {index}", end='\r')
        # Move tensors to GPU
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = generator(x)  # Get prediction from Generator
            D_real = discriminator(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = discriminator(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss+D_fake_loss)/2

        discriminator.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = discriminator(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y)*config.L1_LAMBDA
            G_loss = G_fake_loss + L1
        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


def main():
    discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    generator = Generator().to(config.DEVICE)
    opt_disc = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE)
    opt_gen = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE)
    train_set = CityScapesDataset('data\City\\train')
    train_loader = DataLoader(train_set,
                              batch_size=20,
                              shuffle=True,
                              num_workers=1)

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_set = CityScapesDataset('data\City\\val')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    if config.LOAD_MODEL:
        pass

    for epoch in range(config.NUM_EPOCHS):
        print(epoch, end="\r")
        train_loop(discriminator, generator, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)


if __name__ == '__main__':
    main()

