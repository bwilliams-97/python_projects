import torch
import torch.nn as nn

class DCGenerator(nn.Module):
    """
    Takes in a latent vector and outputs a synthetic image
    that should be in same distribution as original dataset.
    """
    def __init__(self):
        super(DCGenerator, self).__init__()

        self.network = nn.Sequential(
            # Feed latent vector z into convolution
            # args: in_channels, out_channels, kernel_size, stride, padding
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False)
            nn.BatchNorm2d(ngf*8)
            nn.ReLU(True)
            # (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 4, 1, bias=False)
            nn.BatchNorm2d(ngf*4)
            nn.ReLU(True)
            # (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 4, 1, bias=False)
            nn.BatchNorm2d(ngf*2)
            nn.ReLU(True)
            # (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf*2, ngf*8=1, 4, 4, 1, bias=False)
            nn.BatchNorm2d(ngf)
            nn.ReLU(True)
            # (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, num_colours, 4, 4, 1, bias=False)
            nn.BatchNorm2d(ngf*8)
            nn.ReLU(True)
            # num_colours x 64 x 64
        )

    def forward(self, input):
        output = self.network(input)

        return output

class DCDiscriminator(nn.Module):
    """
    Takes in an image and outputs the probability this image
    is from the original dataset.
    """
    def __init__(self):
        super(DCDiscriminator, self).__init__()

        self.network == nn.Sequential(
            # input is num_colours x 64 x 64
            nn.Conv2d(num_colours, ndf, 4, 2, 1, bias=False)
            nn.LeakyReLU(0.2, inplace=True)

            # (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)
            nn.BatchNorm2d(ndf*2)
            nn.LeakyReLU(0.2, inplace=True)

            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)
            nn.BatchNorm2d(ndf*4)
            nn.LeakyReLU(0.2, inplace=True)

            # (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)
            nn.BatchNorm2d(ndf*8)
            nn.LeakyReLU(0.2, inplace=True)

            # (ndf) x  4 x 4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)

        return output.view(-1, 1).squeeze(1)