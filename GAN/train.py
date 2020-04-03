import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

def train(
    generator: torch.nn.Module, 
    discriminator: torch.nn.Module,
    n_epochs: int,
    fixed_noise: torch.tensor,
    train_loader: torch.utils.data.DataLoader,
    latent_size: int
    ):

    criterion = nn.BCELoss()

    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=0.0002)
    generator_optimiser = optim.Adam(generator.parameters(), lr=0.0002)

    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader, 0):
            print(i)
            ###################################
            # Update D
            ###################################
            discriminator.zero_grad()

            real_data = data[0]
            batch_size = real_data.size(0)
            label = torch.ones((batch_size, ))

            output = discriminator(real_data)
            error_real = criterion(output, label)
            error_real.backward()

            # train with fake
            latent_noise = torch.randn(batch_size, latent_size, 1, 1)
            fake = generator(latent_noise)
            output = discriminator(fake.detach())
            error_fake = criterion(output, label)
            error_fake.backward()

            total_error = error_real + error_fake
            discriminator_optimiser.step()

            ###################################
            # Update G
            ###################################
            generator.zero_grad()

            output = discriminator(fake)
            error_gen = criterion(output, label)
            error_gen.backward()

            generator_optimiser.step()

            if i % 100 == 0:
                save_image(real_data,
                        '%s/real_samples.png' % "images",
                        normalize=True)
                fake = generator(fixed_noise)
                save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % ("images", epoch),
                        normalize=True)
