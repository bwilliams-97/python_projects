import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from models import ImageDiscriminator
from utils import set_grad


def train(
    trainloader: DataLoader,
    lego_generator: nn.Module, 
    house_generator: nn.Module,
    lego_discriminator: nn.Module, 
    house_discriminator: nn.Module,
    **kwargs
) -> None:
    adversarial_criterion = nn.BCELoss()
    cycle_criterion = nn.L1Loss()
    cycle_lambda = kwargs["cycle_lambda"]

    learning_rate = kwargs["learning_rate"]
    generator_optimiser = optim.Adam(itertools.chain(lego_generator.parameters(), house_generator.parameters()), lr=learning_rate)
    discriminator_optimiser = optim.Adam(itertools.chain(lego_discriminator.parameters(), house_discriminator.parameters()), lr=learning_rate)

    for epoch in tqdm(range(kwargs["n_epochs"])):
        for i, data in enumerate(trainloader):
            print(f"Epoch: {epoch}, Batch: {i}", end='\r')
            
            lego_images, house_images = data[0], data[1]

            # Optimise generators first
            set_grad([lego_discriminator, house_discriminator], False)
            generator_optimiser.zero_grad()

            house_fake = house_generator(lego_images)
            lego_fake = lego_generator(house_images)

            house_recon = house_generator(lego_fake)
            lego_recon = lego_generator(house_fake)

            # Calculate discriminator output
            house_fake_dis = house_discriminator(house_fake)
            lego_fake_dis = lego_discriminator(lego_fake)

            real_label = torch.ones(house_fake_dis.size())

            house_gen_loss = adversarial_criterion(house_fake_dis, real_label)
            lego_gen_loss = adversarial_criterion(lego_fake_dis, real_label)

            # Calculate cycle consistency loss
            house_cycle_loss = cycle_lambda * cycle_criterion(house_images, house_recon)
            lego_cycle_loss = cycle_lambda * cycle_criterion(lego_images, lego_recon)

            gen_loss = house_gen_loss + lego_gen_loss + house_cycle_loss + lego_cycle_loss
            gen_loss.backward()
            generator_optimiser.step()

            # Now optimise discriminators
            set_grad([lego_discriminator, house_discriminator], True)
            discriminator_optimiser.zero_grad()

            house_fake = torch.tensor(house_fake.data.numpy())
            lego_fake = torch.tensor(lego_fake.data.numpy())

            house_real_dis = house_discriminator(house_images)
            house_fake_dis = house_discriminator(house_fake)
            lego_real_dis = lego_discriminator(lego_images)
            lego_fake_dis = lego_discriminator(lego_fake)

            real_label = torch.ones(house_real_dis.size())
            fake_label = torch.zeros(house_fake_dis.size())

            house_dis_real_loss = adversarial_criterion(house_real_dis, real_label)
            house_dis_fake_loss = adversarial_criterion(house_fake_dis, fake_label)
            lego_dis_real_loss = adversarial_criterion(lego_real_dis, real_label)
            lego_dis_fake_loss = adversarial_criterion(lego_fake_dis, fake_label)

            # Total discriminators losses
            house_dis_loss = (house_dis_real_loss + house_dis_fake_loss) * 0.5
            lego_dis_loss = (lego_dis_real_loss + lego_dis_fake_loss) * 0.5

            # Update discriminators
            house_dis_loss.backward()
            lego_dis_loss.backward()
            discriminator_optimiser.step()

        fake_lego = lego_generator(house_images)
        save_image(fake_lego.detach(),
            '%s/fake_lego_epoch_%03d.png' % (kwargs["output_dir"], epoch))
        fake_house = house_generator(lego_images)
        save_image(fake_house.detach(),
            '%s/fake_house_epoch_%03d.png' % (kwargs["output_dir"], epoch))