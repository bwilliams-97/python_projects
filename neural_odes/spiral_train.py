from typing import Callable

import torch
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt

from models import ODESystem, VanillaODEFunc


class SpiralODE(ODESystem):
    def __init__(self, A: torch.tensor):
        super(SpiralODE, self).__init__(A)

    def forward(self, t: torch.tensor, y: torch.tensor) -> torch.tensor:
        return torch.mm(y**3, self.A)


def generate_points(num_timesteps: int, t_end: int, y_0: torch.tensor, ode_system: ODESystem) -> torch.tensor:
    t = torch.linspace(0, t_end, num_timesteps)
    with torch.no_grad():
        y_true = odeint(ode_system, y_0, t)
    return y_true


def train_step(
    ode_model: VanillaODEFunc, 
    optimiser: torch.optim.Optimizer, 
    loss_function: Callable,
    y_true: torch.tensor,

) -> None:
    optimiser.zero_grad()
    sample_y_0, sample_t, sample_y = get_batch_y()
    y_pred = odeint(ode_model, sample_y_0, sample_t)

    loss = loss_function(y_pred, sample_y)
    loss.backward()
    optimiser.step()


def test_step(
    ode_model: VanillaODEFunc,
    loss_function: Callable,
    y_0: torch.tensor,
    y_true: torch.tensor,
    t: torch.tensor
) -> None:
    with torch.no_grad():
        y_pred = odeint(ode_model, y_0, t)
        loss = loss_function(y_pred, y_true)


def train_model(
    num_iterations: int,
    ode_model: VanillaODEFunc,
    y_0: torch.tensor,
    y_true: torch.tensor,
    t: torch.tensor
) -> None:
    optimiser = torch.optim.Adam(ode_model.parameters())
    loss_function = nn.L1Loss()

    for iteration in range(num_iterations):
        train_step(ode_model, optimiser, loss_function)

        if iteration % 20 == 0:
            test_step(ode_model, loss_function, y_0, y_true, t)


def main():
    ode_model = VanillaODEFunc(state_size = 2)

    num_iterations = 2000
    data_size = 1000

    y_0 = torch.tensor([2., 0.])
    t = torch.linspace(0., 25., data_size)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
    ode_system = SpiralODE(true_A)

    y_true = generate_points(data_size, 25.0, y_0, ode_system)

    train_model(num_iterations, ode_model, y_0, y_true, t)








