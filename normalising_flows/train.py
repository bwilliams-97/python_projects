import argparse
from typing import Union
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import tqdm
from models import VanillaVAE
from torchvision.utils import save_image



def train(train_loader: torch.utils.data.DataLoader, model, **kwargs):
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
            reconstructed, mu, logvar = model(data)

            # Backpropagate loss
            # loss = vae_loss_function(reconstructed, data, mu, logvar)
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

