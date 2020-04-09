# Generative adversarial network
[GANs](https://arxiv.org/pdf/1406.2661.pdf) use two neural networks to train a generative model. 
The first, a _generator_, generates data from some input latent vector. The aim is to generate data in the same distribution as the training data.
The second, a _discriminator_, predicts the likelihood of an input data point belonging to the training set.

## Models
1. DCGAN - Convolutional GAN with encoder/decoder structure based on PyTorch [examples](https://github.com/pytorch/examples).
2. Conditional GAN - model incorporates class label, to allow generation of new image with this label. Model uses linear units rather than convolutions.
