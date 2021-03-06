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
    
    Parameters:
        reconstructed_x (torch.tensor): Output of VAE.
        x (torch.tensor): Input to VAE.
        mu (torch.tensor): Mean vector of latent distribution of model.
        logvar (torch.tensor): Log variance of latent distribution of model

    Returns:
        torch.tensor: Scalar loss 
    """
    # Variational lower bound, which we estimate by sampling from latent space. We want to calculate the expectation
    # of log(p(x_i | z))] over the approximate posterior and do so on the pixel level 
    # using the sigmoid outputs of the model (pixels are binary).
    bce_loss = F.binary_cross_entropy(reconstructed_x, x.view(-1, 784), reduction='sum')

    # KL divergence time is analytically integrable in this way
    # when both prior p_theta(z) and approximate posterior of z are Gaussian.
    kl_divergence = -0.5 * torch.sum(-torch.exp(logvar) - mu.pow(2) + 1.0 + logvar)

    return bce_loss + kl_divergence

def encode_one_hot(n_classes: int, class_idx: torch.tensor) -> torch.tensor:
    """
    Takes in numeric class labels and returns one-hot encoding tensor.

    Parameters:
        n_classes (int): Number of unique classes.
        class_idx (torch.tensor): (D x 1) tensor of class labels.
    """
    assert class_idx.shape[1] == 1
    assert torch.max(class_idx).item() < n_classes

    one_hot_encoding = torch.zeros(class_idx.shape[0], n_classes)
    one_hot_encoding.scatter_(1, class_idx.data, 1)

    return one_hot_encoding

def train(train_loader: torch.utils.data.DataLoader, model: Union[VanillaVAE], **kwargs):
    """
    Train model and generate sample images after each epoch.

    Parameters:
        train_loader (DataLoader): Input dataset (E.g. MNIST).
        model (Union[VanillaVAE]): VAE model derived from VAEBaseClass.

    Returns:
        None
    """

    optimizer = torch.optim.Adam(model.parameters(), lr = kwargs["learning_rate"])
    n_classes = kwargs["n_classes"]

    for epoch in tqdm.trange(kwargs["epochs"]):
        model.train()

        train_loss = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            if kwargs["model_type"] == "vanilla":
                reconstructed, mu, logvar = model(data)
            elif kwargs["model_type"] == "conditional":
                reconstructed, mu, logvar = model(data, encode_one_hot(n_classes, labels.view(-1,1)))

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
        save_epoch_image(model, epoch, kwargs["device"], kwargs["model_type"], n_classes)

    print("Epoch: {}, Average loss: {:.4g}".format(
        epoch, train_loss/len(train_loader.dataset)))

def save_epoch_image(model: Union[VanillaVAE], epoch: int, device: torch.device, model_type: str, n_classes: int):
    """
    Generate sample images and save to file.
    Parameters:
        model (Union[VanillaVAE]): VAE that generates samples.
        epoch (int): Optimisation iteration of training.
        device (torch.device): Device to train on (cpu or cuda)
        model_type (str): Type of VAE.
        n_classes (int): Number of possible label classes.
    Returns:
        None
    """
    model.eval()
    # Generate sample images
    with torch.no_grad():
        sample = torch.randn(32, model.latent_size).to(device)
        import pdb; pdb.set_trace()
        if model_type == "vanilla":
            decoded_sample = model.decode(sample).cpu()
        elif model_type == "conditional":
            class_labels = torch.randint(0, n_classes, (32, 1)).to(device)
            one_hot_encoding = encode_one_hot(n_classes, class_labels)
            decoded_sample = model.decode(torch.cat((sample, one_hot_encoding), dim=1))

        save_image(decoded_sample.view(32, 1, 28, 28),
                    'results/image_' + str(epoch) + '.png')

