import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

def encode_one_hot(n_classes: int, class_idx: torch.tensor) -> torch.tensor:
    """
    Takes in numeric class labels and returns one-hot encoding tensor.

    Parameters:
        n_classes (int): Number of unique classes.
        class_idx (torch.tensor): (D x 1) tensor of class labels.
    """
    assert class_idx.shape[1] == 1
    assert torch.max(class_idx).item() < n_classes

    one_hot_encoding = torch.zeros(class_idx.shape[0], n_classes)
    one_hot_encoding.scatter_(1, class_idx.data, 1)

    return one_hot_encoding

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
        n_classes (int): Number of unique image classes.
    """

    criterion = nn.BCELoss()

    learning_rate = kwargs["learning_rate"]
    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=learning_rate)
    generator_optimiser = optim.Adam(generator.parameters(), lr=learning_rate)

    model_type = kwargs["model_type"]

    for epoch in tqdm(range(kwargs["n_epochs"])):
        for i, (data, img_labels) in enumerate(train_loader):
            discriminator.zero_grad()

            # First run discriminator on real images
            real_imgs = data
            batch_size = real_imgs.size(0)
            label = torch.ones((batch_size, ))

            if model_type == "dcgan":
                real_discriminator_predictions = discriminator(real_imgs)
            elif model_type == "conditional":
                real_discriminator_predictions = discriminator(real_imgs, encode_one_hot(kwargs["n_classes"], img_labels.view(-1,1)))

            # How well does discriminator recognise real images?
            error_real = criterion(real_discriminator_predictions, label)
            error_real.backward()

            # Second run discriminator on fake generated images
            if model_type == "dcgan":
                latent_noise = torch.randn(batch_size, kwargs["latent_size"], 1, 1)
                fake_imgs = generator(latent_noise)

            elif model_type == "conditional":
                latent_noise = torch.randn(batch_size, kwargs["latent_size"])
                latent_class_labels = encode_one_hot(
                    kwargs["n_classes"], torch.randint(0, kwargs["n_classes"], (batch_size, 1))
                )
                fake_imgs = generator(latent_noise, latent_class_labels)

            # Overwrite label, as images are now fake.
            label.fill_(0.)
            if model_type == "dcgan":
                fake_descriminator_predictions = discriminator(fake_imgs.detach())
            elif model_type == "conditional":
                fake_descriminator_predictions = discriminator(fake_imgs.detach(), latent_class_labels.detach())

            # How well does discriminator recognise fake images?
            error_fake = criterion(fake_descriminator_predictions, label)
            error_fake.backward()

            # Update discriminator to improve at predicting which images are real and fake
            total_error = error_real + error_fake
            discriminator_optimiser.step()

            generator.zero_grad()

            # Now run discriminator on fake images again
            if model_type == "dcgan":
                fake_descriminator_predictions = discriminator(fake_imgs)
            elif model_type == "conditional":
                fake_descriminator_predictions = discriminator(fake_imgs, latent_class_labels)

            # Overwrite label, as we want discriminator to falsely classify generated images
            label.fill_(1.)
            # How well does generator fool discriminator?
            error_gen = criterion(fake_descriminator_predictions, label)
            error_gen.backward()

            # Update generator to improve at fooling discriminator
            generator_optimiser.step()
            
            if i % 100 == 0:
                save_image(real_imgs,
                        '%s/real_samples.png' % "images",
                        normalize=True)

                if model_type == "dcgan":
                    fake = generator(kwargs["fixed_noise"])
                    save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % ("images", epoch),
                        normalize=True)

                elif model_type == "conditional":
                    save_class_labels = encode_one_hot(
                        kwargs["n_classes"], torch.randint(0, kwargs["n_classes"], (batch_size, 1))
                    )
                    fake = generator(kwargs["fixed_noise"], save_class_labels)
                    save_image(fake.detach().view(batch_size, 1, 28, 28),
                        '%s/fake_samples_epoch_%03d.png' % ("images", epoch),
                        normalize=True)

                
