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


class CGImageDataset(Dataset):
    def __init__(self, images: List[np.ndarray], crop_images: bool=True, image_size: int = 512):
        """
        @param images: List of images represented as pixel values in numpy arrays.
        @param crop_images: If True, images will be cropped to constant size.
        @param image_size: Size of image to crop to.
        """
        # Image dimensions are [H x W x C]
        if crop_images:
            self._images = self.crop_images(images, image_size)
        else:
            self._images = images

    def crop_images(self, images: List[np.ndarray], image_size: int) -> List[np.ndarray]:
        """
        Crop all images in dataset to same square size.
        @param images: List of images to crop.
        @param image_size: Side length of square (pixels).
        """
        processed_images = []
        for image in images:
            x_dim, y_dim = image.shape[0], image.shape[1]
            # If image is too small, ignore
            if np.min((x_dim, y_dim)) < image_size:
                continue
            # Otherwise select central portion
            else:
                start_x = x_dim // 2 - image_size // 2
                start_y = y_dim // 2 - image_size // 2
                processed_images.append(image[start_x : start_x + image_size, start_y : start_y + image_size])

        return processed_images

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
    house_images = CGImageDataset(house_images)


if __name__ == "__main__":
    main()