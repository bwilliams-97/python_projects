from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

class VAEBaseClass(nn.Module):
    """
    The VAE approximates the posterior over some latent variables by using an encoder and decoder.
    
    The generative decoder can produce outputs via p(x|z)p(z), with p(z) being a distribution that
    is simple to sample from and p(x|z) modelled by the decoder network.

    The encoder represents q(z|x). We constrain this posterior distribution to be of the Gaussian family,
    but with mu and the log variance some complex function of the inputs that is learned by the encoder.

    By reparameterising we are able to backpropagate through all of the component layers.
    """
    def __init__(self, **kwargs):
        super(VAEBaseClass, self).__init__()

        self.input_size = kwargs["input_size"]
        self.latent_size = kwargs["latent_size"]
    
    def encode(self):
        """
        Encoding layer maps from input space into the latent space.
        """
        pass

    def reparameterise(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Sample from the latent distribution 
        """
        eps = torch.randn(self.latent_size)
        std = torch.exp(0.5*logvar)
        return mu + eps*std

    def decode(self):
        """
        Decoding layer maps from latent space into the input space.
        """
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

class ConditionalVAE(VanillaVAE):
    def __init__(self, **kwargs):
        super(ConditionalVAE, self).__init__(**kwargs)
        self.n_classes = kwargs["n_classes"]

        # Overwrite relevant layers to include one-hot encoding of classes.
        self.forward_1 = nn.Linear(self.input_size + self.n_classes, 400)
        self.forward_3 = nn.Linear(self.latent_size + self.n_classes, 400)

    def forward(self, x: torch.tensor, y: torch.tensor) -> Tuple[torch.tensor]:
        # Flatten image
        x = x.view(-1, self.input_size)
        # Concatenate class label with input image
        x = torch.cat((x, y), dim=1)
        # Encode to latent space
        mu, logvar = self.encode(x)
        # Generate samples from latent distribution
        z = self.reparameterise(mu, logvar)
        # Concatenate class label with latent vector
        z = torch.cat((z, y), dim=1)
        # Decode back into input space
        return self.decode(z), mu, logvar