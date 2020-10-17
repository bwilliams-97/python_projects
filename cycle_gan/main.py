from typing import List
from glob import glob
from pathlib import Path
from PIL import Image
import os

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


CYCLE_GAN_PATH = Path(os.path.abspath(__file__)).parent


def ImageDataset(Dataset):
    def __init__(self, images: List[np.ndarray]):
        self._images = images

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        return torch.tensor(self._images[idx])



def load_image_data(directory: Path) -> List[np.ndarray]:
    """
    Load all image files within a given directory.
    @param directory: Directory to load images from.
    @returns List of numpy array representations of images.
    """
    image_files = find_files_with_extension(directory, "jpg")
    image_files.extend(find_files_with_extension("png"))

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

    lego_dataset = ImageDataset(lego_images)
    house_images = ImageDataset(house_images)



if __name__ == "__main__":
    main()