from typing import Callable

import torch
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np

from models import ODESystem, VanillaODEFunc


class SpiralODE(ODESystem):
    def __init__(self, A: torch.tensor):
        super(SpiralODE, self).__init__(A)

    def forward(self, t: torch.tensor, y: torch.tensor) -> torch.tensor:
        return torch.mm(y**3, self.A)


class TrainSpec:
    def __init__(self, data_size: int, batch_timesteps: int, batch_size: int):
        self.data_size = data_size
        self.batch_timesteps = batch_timesteps
        self.batch_size = batch_size


def generate_points(num_timesteps: int, t_end: int, y_0: torch.tensor, ode_system: ODESystem) -> torch.tensor:
    t = torch.linspace(0, t_end, num_timesteps)
    with torch.no_grad():
        y_true = odeint(ode_system, y_0, t)
    return y_true


def get_batch_y(data_size: int, batch_timesteps: int, batch_size: int, y_true: torch.tensor, t: torch.tensor):
    d_points = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time), batch_size, replace=False)))

    y_0_batch = y_true[d_points]
    t_batch = t[:batch_timesteps]
    y_batch = torch.stack([y_true[d_points + i] for i in range(batch_time)], dim=0)

    return y_0_batch, t_batch, y_batch



def train_step(
    ode_model: VanillaODEFunc, 
    optimiser: torch.optim.Optimizer, 
    loss_function: Callable,
    y_true: torch.tensor,
    t: torch.tensor,
    batch_spec: TrainSpec
) -> None:
    optimiser.zero_grad()
    sample_y_0, sample_t, sample_y = get_batch_y(
        batch_spec.data_size, batch_spec.batch_timesteps, batch_spec.batch_size,
        y_true, t
    )
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
    t: torch.tensor,
    batch_spec: TrainSpec
) -> None:
    optimiser = torch.optim.Adam(ode_model.parameters())
    loss_function = nn.L1Loss()

    for iteration in range(num_iterations):
        train_step(ode_model, optimiser, loss_function, y_true, t, batch_spec)

        if iteration % 20 == 0:
            test_step(ode_model, loss_function, y_0, y_true, t)


def main():
    num_iterations = 2000
    data_size = 1000
    batch_timesteps = 10
    batch_size = 1000

    ode_model = VanillaODEFunc(state_size = 2)
    batch_spec = TrainSpec(data_size, batch_timesteps, batch_size)


    y_0 = torch.tensor([2., 0.])
    t = torch.linspace(0., 25., data_size)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
    ode_system = SpiralODE(true_A)

    y_true = generate_points(data_size, 25.0, y_0, ode_system)

    train_model(num_iterations, ode_model, y_0, y_true, t, batch_spec)
