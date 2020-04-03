import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

def train(
    generator: torch.nn.Module, 
    discriminator: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    **kwargs
    ) -> None:
    """
    Training a GAN has two key phases. For each epoch and minibatch:
        1. Run real images (label=1) and fake images from generator network (label=0) through discriminator.
        Optimise discriminator to predict fake and real images.
        2. Run fake images from generator through discriminator.
        Optimise generator to fool discriminator network.

        Optimisation for each network is done separately.
    
    Params:
        generator (torch.nn.Module): Generator network, generates image from input latent vector.
        discriminator (torch.nn.Module): Discriminator network, predicts likelihood of image being
            real.
        train_loader (torch.utils.data.DataLoader): Input PyTorch DataLoader object.

    Keyword args:
        learning_rate (float): Learning rate of both optimisers.
        n_epochs (int): Number of epochs to train for.
        latent_size (int): Size of latent vector generator takes as input.
        fixed_noise (torch.tensor): Fixed noise vector that is used to generate saved images
            for generator assessment.
    """

    criterion = nn.BCELoss()

    learning_rate = kwargs["learning_rate"]
    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=learning_rate)
    generator_optimiser = optim.Adam(generator.parameters(), lr=learning_rate)

    for epoch in tqdm(range(kwargs["n_epochs"])):
        for i, data in enumerate(train_loader, 0):
            discriminator.zero_grad()

            # First run discriminator on real images
            real_imgs = data[0]
            batch_size = real_imgs.size(0)
            label = torch.ones((batch_size, ))

            real_discriminator_predictions = discriminator(real_imgs)
            # How well does discriminator recognise real images?
            error_real = criterion(real_discriminator_predictions, label)
            error_real.backward()

            # Second run discriminator on fake generated images
            latent_noise = torch.randn(batch_size, kwargs["latent_size"], 1, 1)
            fake_imgs = generator(latent_noise)

            # Overwrite label, as images are now fake.
            label.fill_(0.)
            fake_descriminator_predictions = discriminator(fake_imgs.detach())
            # How well does discriminator recognise fake images?
            error_fake = criterion(fake_descriminator_predictions, label)
            error_fake.backward()

            # Update discriminator to improve at predicting which images are real and fake
            total_error = error_real + error_fake
            discriminator_optimiser.step()

            generator.zero_grad()

            # Now run discriminator on fake images again
            fake_descriminator_predictions = discriminator(fake_imgs)

            # Overwrite label, as we want discriminator to falsely classify generated images
            label.fill_(1.)
            # How well does generator fool discriminator?
            error_gen = criterion(fake_descriminator_predictions, label)
            error_gen.backward()

            # Update generator to improve at fooling discriminator
            generator_optimiser.step()
            print(i)
            if i % 100 == 0:
                save_image(real_imgs,
                        '%s/real_samples.png' % "images",
                        normalize=True)
                fake = generator(kwargs["fixed_noise"])
                save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % ("images", epoch),
                        normalize=True)
