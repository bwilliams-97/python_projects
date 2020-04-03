import argparse

import torch
from torchvision import datasets, transforms

from models import DCDiscriminator, DCGenerator
from train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAN on MNIST.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs.")
    parser.add_argument("--latent-size", type=int, default=100,
                        help="Size of latent random vector images are generated from.")
    parser.add_argument("--learning-rate", type=int, default=2e-3,
                        help="Learning rate of optimiser")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device, cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Minibatch size for training.")
    parser.add_argument("--n-generator-filters", type=int, default=64,
                        help="Number of channels used by generator.")
    parser.add_argument("--n-discriminator-filters", type=int, default=64,
                        help="Number of channels used by discriminator.")
    parser.add_argument("--n-colours", type=int, default=1,
                        help="Number of colour channels in images.")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Housekeeping
    torch.manual_seed(0)
    device = torch.device("cuda" if args.device=="cuda" else "cpu") # change to args
    train_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.device=="cuda" else {}# change to args

    # Load dataset
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **train_loader_kwargs
        )

    # Arguments for both models
    n_colours = args.n_colours

    # Generator arguments
    latent_size = args.latent_size
    ngf = args.n_generator_filters

    # Define generator
    generator_network = DCGenerator(latent_size, ngf, n_colours)
    print('Generator: ', generator_network)

    # Fixed noise that will be used to examine generator output in saved images
    fixed_noise = torch.randn(args.batch_size, latent_size, 1, 1)

    # Discriminator arguments
    ndf = args.n_discriminator_filters

    # Define discriminator
    discriminator_network = DCDiscriminator(ndf, n_colours)
    print('Discriminator: ', discriminator_network)
    
    train_kwargs = {
        "n_epochs": args.epochs,
        "latent_size": latent_size,
        "learning_rate": args.learning_rate,
        "fixed_noise": fixed_noise
    }

    train(generator_network, discriminator_network, train_loader, **train_kwargs)

if __name__ == "__main__":
    main()
