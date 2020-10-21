from typing import List

import torch
from torch.utils.data import Dataset
import numpy as np


class CGImageDataset(Dataset):
    def __init__(self, lego_images: List[np.ndarray], house_images: List[np.ndarray], image_size: int = 512):
        """
        @param images: List of images represented as pixel values in numpy arrays.
        @param image_size: Size of image to crop to.
        """
        # Image dimensions are [H x W x C]
        self._lego_images = self.crop_images(lego_images, image_size)
        # Permute to [N x C x H x W]
        self._lego_images = np.moveaxis(self._lego_images, 3, 1)

        # Repeat for house images
        self._house_images = self.crop_images(house_images, image_size)
        # Permute to [N x C x H x W]
        self._house_images = np.moveaxis(self._house_images, 3, 1)

    def crop_images(self, images: List[np.ndarray], image_size: int) -> np.ndarray:
        """
        Crop all images in dataset to same square size.
        @param images: List of images to crop.
        @param image_size: Side length of square (pixels).
        """
        processed_images = []
        for image in images:
            x_dim, y_dim = image.shape[0], image.shape[1]
            # If image is too small, ignore
            if np.min((x_dim, y_dim)) < image_size or len(image.shape) < 3 or image.shape[2] > 3:
                continue
            # Otherwise select central portion
            else:
                start_x = x_dim // 2 - image_size // 2
                start_y = y_dim // 2 - image_size // 2
                processed_images.append(image[start_x : start_x + image_size, start_y : start_y + image_size])

        return np.array(processed_images)

    def __len__(self):
        # Todo: This is a hack to keep same length. Work out a better way of doing this!
        return np.min([self._lego_images.shape[0], self._house_images.shape[0]])

    def __getitem__(self, idx):
        lego_image = torch.tensor(self._lego_images[idx], dtype=torch.float32)
        house_image = torch.tensor(self._house_images[idx], dtype=torch.float32)
        return lego_image, house_image