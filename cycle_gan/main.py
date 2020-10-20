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
    lego_image_dir = CYCLE_GAN_PATH / Path("lego_houses")
    house_image_dir = CYCLE_GAN_PATH / Path("houses")

    lego_images = load_image_data(lego_image_dir)
    house_images = load_image_data(house_image_dir)

    lego_dataset = CGImageDataset(lego_images)
    lego_dataloader = DataLoader(lego_dataset, batch_size=32, shuffle=True)

    house_dataset = CGImageDataset(house_images)
    house_dataloader = DataLoader(house_dataset, batch_size=32, shuffle=True)

    train(lego_dataloader, house_dataloader, learning_rate=1e-3, n_epochs=10)


if __name__ == "__main__":
    main()