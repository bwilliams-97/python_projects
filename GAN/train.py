

for epoch in range(n_epochs):
    for i, data in enumerate(dataloader, 0):
        ###################################
        # Update D
        ###################################
        discriminator.zero_grad()

        output = discriminator(real_data)
        error_real = criterion(output, label)
        error_real.backward()

        # train with fake
        latent_noise = torch.randn(batch_size, nz, 1, 1)
        fake = generator(latent_noise)
        output = discriminator(fake.detach())
        error_fake = criterion(output, label)
        error_fake.backward()

        total_error - error_real + error_fake
        discriminator_optimiser.step()

        ###################################
        # Update G
        ###################################
        generator.zero_grad()

        output = discriminator(fake)
        error_gen = criterion(output, label)
        error_gen.backward()

        generator_optimiser.step()
