import torch
import torch.nn as nn


def VanillaODEFunc(nn.Module):
    def __init__(self, state_size: int):
        super(ODEFunc, self).__init__()

        self.basic_net = nn.Sequential(
            nn.Linear(state_size, 50),
            nn.Tanh(),
            nn.Linear(50, state_size)
        )

    def forward(self, t: torch.tensor, y: torch.tensor) -> torch.tensor:
        return self.basic_net(y**3)



class ODESystem(nn.Module):
    def __init__(self, A: torch.tensor):
        super(ODESystem, self).__init__()
        self.A = A

    def forward(self, t: torch.tensor, y: torch.tensor) -> torch.tensor:
        pass