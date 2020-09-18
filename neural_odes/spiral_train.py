from typing import Callable, Tuple
import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models import ODESystem, VanillaODEFunc


class SpiralODE(ODESystem):
    """
    Torch module that generates trajectory based on ODE.
    """
    def __init__(self, A: torch.tensor):
        super(SpiralODE, self).__init__(A)

    def forward(self, t: torch.tensor, y: torch.tensor) -> torch.tensor:
        return torch.mm(y**3, self.A)


class BatchSpec:
    """
    Class to store information for batch sampling.
    """
    def __init__(self, total_timesteps: int, batch_timesteps: int, batch_size: int):
        self.total_timesteps = total_timesteps
        self.batch_timesteps = batch_timesteps
        self.batch_size = batch_size


def parse_args() -> argparse.Namespace:
    """
    Parse arguments of script
    """
    parser = argparse.ArgumentParser(description='Generate network.')

    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--total_timesteps", type=int, default=1000, help="Total number timesteps.")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size to use for training.")
    parser.add_argument("--time_end", type=float, default=25.0, 
                        help="End time of trajectory in seconds.")
    parser.add_argument("--batch_timesteps", type=int, default=10, 
                        help="Number of timesteps to sample in each batch.")
    parser.add_argument("--output_dir", type=str, default="fig", 
                        help="Directory to save outputs to e.g. Spiral figures")

    args = parser.parse_args()
    return args


def generate_points(num_timesteps: int, t_end: int, y_0: torch.tensor, ode_system: ODESystem
                ) -> Tuple[torch.tensor, torch.tensor]:
    """
    Generate points from ODE system using a set number timesteps and an initial condition y_0.
    @param num_timesteps: Number of timesteps to generate points from.
    @param t_end: Length of time interval to get points from.
    @param y_0: Initial condition of system.
    @param ode_system: ODE system to generate points with via odeint function.
    """
    t = torch.linspace(0, t_end, num_timesteps)
    with torch.no_grad():
        y_true = odeint(ode_system, y_0, t)
    return t, y_true


def get_batch_y(total_timesteps: int, batch_timesteps: int, batch_size: int, y_true: torch.tensor, t: torch.tensor):
    """
    Get a particular batch by selecting subsections of the trajectory in y_true.
    @param total_timesteps: Total number of timesteps in trajectory.
    @param batch_timesteps: Number of timesteps to sample per batch.
    @param batch_size: Number of sub-sections to sample per batch.
    @param y_true: Actual position of trajectory at each timestep.
    @param t: Torch tensor of timesteps.
    """
    # Select a random set of starting points from which to get system trajectory.
    initial_timesteps = torch.from_numpy(
        np.random.choice(np.arange(total_timesteps - batch_timesteps), batch_size, replace=False)
    ).type(torch.long)

    # Set of initial conditions for the batch
    y_0_batch = y_true[initial_timesteps]
    # Set of timesteps to use (currently zero indexed).
    t_batch = t[:batch_timesteps]
    # Select subset of trajectory for each batch starting from initial_timesteps and extending for batch_timesteps.
    y_batch = torch.stack([y_true[initial_timesteps + i] for i in range(batch_timesteps)], dim=0)

    return y_0_batch, t_batch, y_batch


def train_step(ode_model: VanillaODEFunc, optimiser: torch.optim.Optimizer, loss_function: Callable,
               y_true: torch.tensor, t: torch.tensor, batch_spec: BatchSpec) -> None:
    """
    Carry out a single training step by obtaining a batch sample of data points, feeding them through the ODE 
    method and backpropagating loss.
    @param ode_model: Torch Module used to parameterise the ODE system.
    @param optimiser: Torch optimiser instance (E.g. Adam, SGD).
    @param loss_function: Torch loss function to use.
    @param y_true: True ODE output for a particular timestep.
    @param t: Set of timepoint indices.
    """
    optimiser.zero_grad()
    # Get batch sample from trajectory.
    sample_y_0, sample_t, sample_y = get_batch_y(
        batch_spec.total_timesteps, batch_spec.batch_timesteps, batch_spec.batch_size,
        y_true, t
    )
    # "Forward" pass through ODE model to predict trajectory
    y_pred = odeint(ode_model, sample_y_0, sample_t)

    loss = loss_function(y_pred, sample_y)
    loss.backward()
    optimiser.step()


def test_step(ode_model: VanillaODEFunc, loss_function: Callable, y_0: torch.tensor, y_true: torch.tensor,
              t: torch.tensor, epoch: int, output_ax: Axes) -> None:
    """
    Plot ODE output based on a given initial point (y_0) and known system dynamics (y_true). Save figure.
    @param ode_model: Torch Module used to parameterise the ODE system.
    @param loss_function: Torch loss function to use.
    @param y_0: Initial condition of system.
    @param y_true: True state of system at each timestep.
    @param t: Tensor of timesteps.
    @param epoch: Number of current epoch.
    @param output_ax: Axes instance to plot trajectory on.
    """
    with torch.no_grad():
        y_pred = odeint(ode_model, y_0, t)
        loss = loss_function(y_pred, y_true)
        output_ax.clear()
        output_ax.plot(y_pred[:, 0, 0], y_pred[:, 0, 1], 'r', y_true[:, 0, 0], y_true[:, 0, 1], 'b')
        plt.draw()
        plt.savefig(f"fig/iter_{epoch}.png")
        plt.pause(0.001)

        print(f"epoch: {epoch}, Loss: {loss}")


def train_model(num_epochs: int, ode_model: VanillaODEFunc, y_0: torch.tensor, y_true: torch.tensor,
                t: torch.tensor, batch_spec: BatchSpec) -> None:
    """
    Train model for given number epochs to learn system dynamics.
    @param ode_model: Torch Module used to parameterise the ODE system.
    @param optimiser: Torch optimiser instance (E.g. Adam, SGD).
    @param loss_function: Torch loss function to use.
    @param y_true: True ODE output for a particular timestep.
    @param t: Set of timepoint indices.
    @param batch_spec: Specification of batch to use (BatchSpec instance).
    """
    optimiser = torch.optim.Adam(ode_model.parameters())
    loss_function = nn.L1Loss()

    fig = plt.figure()
    ax_traj = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    for epoch in tqdm(range(num_epochs)):
        train_step(ode_model, optimiser, loss_function, y_true, t, batch_spec)

        if epoch % 20 == 0:
            test_step(ode_model, loss_function, y_0, y_true, t, epoch, ax_traj)


def main():
    args = parse_args()

    # Generate output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Declare ODE model to use
    ode_model = VanillaODEFunc(state_size = 2)
    # Specify how to generate batches
    batch_spec = BatchSpec(args.total_timesteps, args.batch_timesteps, args.batch_size)

    # Initial condition
    y_0 = torch.tensor([[2., 0.]])
    # State matrix
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
    # ODE system module
    ode_system = SpiralODE(true_A)

    # Generate points along trajectory starting from initial condition
    t, y_true = generate_points(args.total_timesteps, args.time_end, y_0, ode_system)

    # Train model to learn system dynamics
    train_model(args.num_epochs, ode_model, y_0, y_true, t, batch_spec)


if __name__ == "__main__":
    main()