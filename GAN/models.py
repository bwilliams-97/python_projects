import torch
import torch.nn as nn

class DCGenerator(nn.Module):
    """
    Takes in a latent vector and outputs a synthetic image
    that should be in same distribution as original dataset.
    """
    def __init__(self, latent_size, ngf, n_colours):
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

    def forward(self, input):
        output = self.network(input)
        
        return output

class DCDiscriminator(nn.Module):
    """
    Takes in an image and outputs the probability this image
    is from the original dataset.
    """
    def __init__(self, ndf, n_colours):
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

    def forward(self, input):
        output = self.network(input)
        
        return output.view(-1, 1).squeeze(1)