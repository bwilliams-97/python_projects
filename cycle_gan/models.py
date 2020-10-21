import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    """
    Double convolutional block (single U-Net unit).
    Twos sets of conv2d + batch_norm + ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: bool=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x) -> torch.tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Single down U-Net unit (downsample with maxpool + double convolutional block).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Single upsample block in U-Net.
    Upsample with bilinear upsampling, concatenate x2 and x1 (x1 comes from down path) 
    and run through double convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
        x1 = self.up(x1)

        # Add padding to ensure same shape tensors.
        diffH = x2.size()[2] - x1.size()[2]
        diffW = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffH // 2, diffH - diffH // 2,
                        diffW // 2, diffW - diffW // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ImageGenerator(nn.Module):
    def __init__(self, image_size: int):
        super(ImageGenerator, self).__init__()
        self.image_size = image_size

        self.start_conv = DoubleConv(3, 8)
        self.down_1 = Down(8, 16)
        self.down_2 = Down(16, 32)
        self.down_3 = Down(32, 32)

        self.up_1 = Up(64, 16)
        self.up_2 = Up(32, 8)
        self.up_3 = Up(16, 3)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x1 = self.start_conv(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        x = self.up_1(x4, x3)
        x = self.up_2(x, x2)
        x = self.up_3(x, x1)

        return x


# Define discriminator model
class ImageDiscriminator(nn.Module):
    def __init__(self, image_size: int):
        super(ImageDiscriminator, self).__init__()

        self.conv_0 = nn.Conv2d(3, 8, kernel_size=5)
        self.bn_0 = nn.BatchNorm2d(8)

        self.conv_1 = nn.Conv2d(8, 16, kernel_size=5)
        self.bn_1 = nn.BatchNorm2d(16)

        self.conv_2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn_2 = nn.BatchNorm2d(32)

        self.conv_activation = nn.LeakyReLU(0.2)
        self.pooling = nn.MaxPool2d(4)

        if image_size == 512:
            self.intermediate_size = 1152
        else:
            raise ValueError("Must implement image intermediate size")

        self.linear_0 = nn.Linear(self.intermediate_size, int(np.sqrt(self.intermediate_size)))
        self.linear_1 = nn.Linear(int(np.sqrt(self.intermediate_size)), 1)

        self.linear_activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, input: torch.tensor) -> torch.tensor:
        # 3 convolutional layers
        x = self.pooling(self.conv_activation(self.bn_0(self.conv_0(input))))
        x = self.pooling(self.conv_activation(self.bn_1(self.conv_1(x))))
        x = self.pooling(self.conv_activation(self.bn_2(self.conv_2(x))))

        x = x.view(x.shape[0], -1)
 
        # 2 linear layers
        x = self.linear_activation(self.linear_0(x))
        x = self.output_activation(self.linear_1(x))

        return x


