# Generative adversarial network
## Model
[GANs](https://arxiv.org/pdf/1406.2661.pdf) use two neural networks to train a generative model. 
The first, a _generator_, generates data from some input latent vector. The aim is to generate data in the same distribution as the training data.
The second, a _discriminator_, predicts the likelihood of an input data point belonging to the training set.

## Code
DCGAN implementation is based on the DCGAN structure found in PyTorch [examples](https://github.com/pytorch/examples).
