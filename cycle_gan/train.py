import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from models import ImageDiscriminator


def train(
    lego_trainloader: DataLoader,
    house_trainloader: DataLoader,
    **kwargs
) -> None:
    criterion = nn.BCELoss()

    discrim = ImageDiscriminator(512)

    learning_rate = kwargs["learning_rate"]
    discriminator_optimiser = optim.Adam(discrim.parameters(), lr=learning_rate)
    # generator_optimiser = optim.Adam(generator.parameters(), lr=learning_rate)

    for epoch in tqdm(range(kwargs["n_epochs"])):
        for i, lego_image in enumerate(lego_trainloader):
            discrim(lego_image)

    # For each epoch, for each batch
        # Create normal images from lego
        # Run normal through discriminator
        # Optimise
        # Run lego through discriminator
        # Optimise
        # Optimise generator based on fake normal images

        # Do same for lego to normal

        # Get L1 distance from normal image recovery
        # Get L1 distance from lego image recovery
        # Optimise