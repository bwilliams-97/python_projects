import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import tqdm
from models import VanillaVAE

def vae_loss_function(x: torch.tensor, reconstructed_x: torch.tensor, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:

    bce_loss = F.binary_cross_entropy(reconstructed_x, x.view(-1)).sum()

    kl_divergence = -0.5 * torch.sum(-torch.exp(logvar) - mu.pow(2) + 1.0 + log_var)

    return bce_loss + kl_divergence

def train(train_loader, model: VanillaVAE, **kwargs):

    optimizer = torch.optim.Adam(model.parameters(), lr = kwargs["learning_rate"])

    for epoch in tqdm.range(kwargs["epochs"]):
        model.train()

        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, logvar = model(data)

            # Backpropagate loss
            loss = vae_loss_function(reconstructed, data, mu, logvar)
            loss.backward

            train_loss += loss.item()

            optimizer.step()

    print("Epoch: {}, Average loss: {:.4g}".format(
        epoch, train_loss/len(train_loader.dataset)))



