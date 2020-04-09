import torch
import torch.nn as nn
from torch.nn import functional as F

class DCGenerator(nn.Module):
    """
    Takes in a latent vector and outputs a synthetic image
    that should be in same distribution as original dataset.
    """
    def __init__(self, latent_size: int, ngf: int, n_colours: int):
        super(DCGenerator, self).__init__()

        self.network = nn.Sequential(
            # Feed latent vector z into convolution
            # args: in_channels, out_channels, kernel_size, stride, padding
            nn.ConvTranspose2d(latent_size, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # (ngf*4) x 4 x 4

            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # (ngf*2) x 7 x 7

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # # (ngf) x 14 x 14
            
            nn.ConvTranspose2d(ngf, n_colours, 4, 2, 1, bias=False),
            nn.Tanh()
            # n_colours x 28 x 28
        )

    def forward(self, z: torch.tensor):
        output = self.network(z)
        
        return output

class DCDiscriminator(nn.Module):
    """
    Takes in an image and outputs the probability this image
    is from the original dataset.
    """
    def __init__(self, ndf: int, n_colours: int):
        super(DCDiscriminator, self).__init__()

        self.network = nn.Sequential(
            # input is n_colours x 28 x 28
            nn.Conv2d(n_colours, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 7 x 7
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4) x  3 x 3
            nn.Conv2d(ndf*4, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input: torch.tensor):
        output = self.network(input)
        
        return output.view(-1, 1).squeeze(1)

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_size: int, n_classes: int, input_size: int):
        super(ConditionalGenerator, self).__init__()

        self.input_size = input_size

        self.forward_1 = nn.Linear(latent_size + n_classes, 400)
        self.forward_2 = nn.Linear(400, self.input_size)

    def forward(self, z: torch.tensor, label: torch.tensor):
        # Concatenate class labels with latent vector
        z = torch.cat((z, label), dim=1)

        hidden_1 = F.relu(self.forward_1(z))
        return torch.sigmoid(self.forward_2(hidden_1))

class ConditionalDiscriminator(nn.Module):
    def __init__(self, n_classes: int, input_size: int):
        super(ConditionalDiscriminator, self).__init__()

        self.input_size = input_size

        self.forward_1 = nn.Linear(self.input_size + n_classes, 400)
        self.forward_2 = nn.Linear(400, 32)
        self.forward_3 = nn.Linear(32, 1)

    def forward(self, x: torch.tensor, label: torch.tensor):
        # Flatten image
        x = x.view(-1, self.input_size)
        # Concatenate class label with input image
        x = torch.cat((x, label), dim=1)

        hidden_1 = F.relu(self.forward_1(x))
        hidden_2 = F.relu(self.forward_2(hidden_1))
        return torch.sigmoid(self.forward_3(hidden_2))