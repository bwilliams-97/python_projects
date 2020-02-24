from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

class VAEBaseClass(nn.Module):
    def __init__(self, **kwargs):
        super(VAEBaseClass, self).__init__()

        self.input_size = kwargs["input_size"]
        self.latent_size = kwargs["latent_size"]
    
    def encode(self):
        pass

    def reparameterise(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        eps = torch.randn(self.latent_size)
        std = torch.exp(0.5*logvar)
        return mu + eps*std

    def decode(self):
        pass

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor]:
        # Flatten input image and encode to latent space
        mu, logvar = self.encode(x.view(-1, self.input_size))
        # Generate samples from latent distribution
        z = self.reparameterise(mu, logvar)
        # Decode back into input space
        return self.decode(z), mu, logvar


class VanillaVAE(VAEBaseClass):
    def __init__(self, **kwargs):
        super(VanillaVAE, self).__init__(**kwargs)
        
        self.forward_1 = nn.Linear(self.input_size, 400)
        self.forward_21 = nn.Linear(400, self.latent_size)
        self.forward_22 = nn.Linear(400, self.latent_size)
        self.forward_3 = nn.Linear(self.latent_size, 400)
        self.forward_4 = nn.Linear(400, self.input_size)

    def encode(self, x: torch.tensor) -> Tuple[torch.tensor]:
        hidden_1 = F.relu(self.forward_1(x))
        return self.forward_21(hidden_1), self.forward_22(hidden_1)

    def decode(self, z: torch.tensor) -> torch.tensor:
        hidden_3 = F.relu(self.forward_3(z))
        return torch.sigmoid(self.forward_4(hidden_3))