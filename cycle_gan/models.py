import torch
import torch.nn as nn
import numpy as np

# Define generator model for lego -> normal domain

# Define generator model for normal -> lego domain

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


