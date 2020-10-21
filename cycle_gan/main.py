from typing import List
from glob import glob
from pathlib import Path
from PIL import Image
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from data_io import CGImageDataset
from train import train
from models import ImageGenerator, ImageDiscriminator


CYCLE_GAN_PATH = Path(os.path.abspath(__file__)).parent


def load_image_data(directory: Path) -> List[np.ndarray]:
    """
    Load all image files within a given directory.
    @param directory: Directory to load images from.
    @returns List of numpy array representations of images.
    """
    image_files = find_files_with_extension(directory, "jpg")
    image_files.extend(find_files_with_extension(directory, "png"))

    image_list = []
    
    print(f"Loading image files from directory {directory.name}...")
    for image_file in tqdm(image_files):
        im = Image.open(image_file)
        image_list.append(np.asarray(im))

    return image_list


def find_files_with_extension(directory: Path, extension: str) -> List[str]:
    """
    Find all files in a given directory with chosen file extension.
    @param directory: Absolute path of directory to look for files in.
    @param extension: File extension (e.g. jpg).
    @returns list of file paths of interest.
    """
    # Template for filepath, some_directory\\some_filepath.extension
    ext_file_path = ''.join([str(directory), f"\\*.{extension}"])
    
    return glob(ext_file_path)


def main():
    image_size = 512

    output_dir = CYCLE_GAN_PATH / Path("output_img")
    os.makedirs(output_dir, exist_ok=True)

    lego_image_dir = CYCLE_GAN_PATH / Path("lego_houses")
    house_image_dir = CYCLE_GAN_PATH / Path("houses")    

    lego_images = load_image_data(lego_image_dir)
    house_images = load_image_data(house_image_dir)

    dataset = CGImageDataset(lego_images, house_images, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    lego_generator = ImageGenerator(image_size=image_size)
    house_generator = ImageGenerator(image_size=image_size)

    lego_discriminator = ImageDiscriminator(image_size=image_size)
    house_discriminator = ImageDiscriminator(image_size=image_size)

    cycle_gan_networks = {
        "lego_generator": lego_generator,
        "house_generator": house_generator,
        "lego_discriminator": lego_discriminator,
        "house_discriminator": house_discriminator
    }

    train(dataloader, lego_generator, house_generator, lego_discriminator, house_discriminator,
          learning_rate=1e-3, n_epochs=10, cycle_lambda=1e-3, output_dir=output_dir)


if __name__ == "__main__":
    main()