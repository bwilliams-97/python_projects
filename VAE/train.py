import argparse
from typing import Union
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import tqdm
from models import VanillaVAE
from torchvision.utils import save_image

def vae_loss_function(reconstructed_x: torch.tensor, x: torch.tensor, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
    """
    Calculate loss for the VAE. This is a combination of the likelihood and the KL divergence.
    """
    bce_loss = F.binary_cross_entropy(reconstructed_x, x.view(-1, 784), reduction='sum')

    kl_divergence = -0.5 * torch.sum(-torch.exp(logvar) - mu.pow(2) + 1.0 + logvar)

    return bce_loss + kl_divergence

def train(train_loader: torch.utils.data.DataLoader, model: Union[VanillaVAE], **kwargs):

    optimizer = torch.optim.Adam(model.parameters(), lr = kwargs["learning_rate"])

    for epoch in tqdm.trange(kwargs["epochs"]):
        model.train()

        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, logvar = model(data)

            # Backpropagate loss
            loss = vae_loss_function(reconstructed, data, mu, logvar)
            loss.backward()

            train_loss += loss.item()

            optimizer.step()

            if batch_idx % 10 == 0:
                print("Epoch: {} Batch: [{}/{}], Train loss: {:.4g}".format(
                    epoch, batch_idx*len(data), len(train_loader.dataset), loss.item()/len(data)
                ))

        # Write model test output for this epoch
        save_epoch_image(model, epoch, kwargs["device"])

    print("Epoch: {}, Average loss: {:.4g}".format(
        epoch, train_loss/len(train_loader.dataset)))

def save_epoch_image(model: Union[VanillaVAE], epoch: int, device: torch.device):
    model.eval()
    # Generate sample images
    with torch.no_grad():
        sample = torch.randn(32, model.latent_size).to(device)
        decoded_sample = model.decode(sample).cpu()
        save_image(decoded_sample.view(32, 1, 28, 28),
                    'results/image_' + str(epoch) + '.png')

