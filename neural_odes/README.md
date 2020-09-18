# Neural ODEs

Implements spiral reconstruction demo as described in [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366.pdf). Rather than learning a relationship between individual samples as with an RNN, the aim of a Neural ODE is to learn the underlying derivative that determines state transitions. Full details can be seen in the original paper.

## Basic demo steps
1. Generate set of points for a state-space system based on an initial condition and state matrix (making use of [torchdiffeq](https://github.com/rtqichen/torchdiffeq) odeint function). The system dynamics must be contained within a PyTorch Module (forward pass implements state space description).
2. Initialise a PyTorch model, that will parameterise the derivative.
3. Train model with odeint function, replacing true system with our neural parameterisation. Each batch is a different subset of the original state trajectory.
4. Backprop loss as normal. This is done by solving another ODE backwards in time using the adjoint sensitivity method (see paper).
5. The test function plots the model trajectory vs the true trajectory each epoch for comparison.

Currently only the VanillaODEFunc is implemented, but deeper PyTorch models could be used.

## Interest in timeseries modelling
Neural ODEs have no requirement for data to arrive at fixed intervals, which removes the need for resampling (as with current timeseries modelling with RNN/CNN based models). Watch this space...