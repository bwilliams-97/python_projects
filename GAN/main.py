import torch
from torchvision import datasets, transforms

from models import DCDiscriminator, DCGenerator
from train import train

device="cuda"
batch_size = 128

# Housekeeping
torch.manual_seed(0)
device = torch.device("cuda" if device=="cuda" else "cpu") # change to args
train_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if device=="cuda" else {}# change to args

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **train_loader_kwargs
    )

latent_size=100
ngf=64
ndf=64
num_colours=1
batch_size=64

generator_network = DCGenerator(latent_size, ngf, num_colours)
print('Generator: ', generator_network)

discriminator_network = DCDiscriminator(ndf, num_colours)
print('Discriminator: ', discriminator_network)

fixed_noise = torch.randn(batch_size, latent_size, 1, 1)

train(generator_network, discriminator_network, 500, fixed_noise, train_loader, latent_size)
