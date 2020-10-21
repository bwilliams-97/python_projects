from typing import List

import torch.nn as nn

def set_grad(nets: List[nn.Module], requires_grad=False) -> None:
    """
    Freeze or activate parameters in a given list of torch modules.
    """
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad