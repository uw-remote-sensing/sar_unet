from typing import Tuple

import torch
import os
from torch.utils.data import Dataset
from skimage.transform import resize
import numpy as np


class LakesSeg(Dataset):
    """
    Dataset class that lets you load image and mask pairs for segmentation.
    """
    def __init__(self, data_dir: str, split: str, scale: int = 4):
        """
        :param data_dir: Path to where imgs/ and masks/ directories are located.
        :param split: Training or testing set.  String should match the txt file name containing image names.
        :param scale: Amount down-sampling. Scale of one leaves images as they are.
        """
        self.data_dir = data_dir
        self.scale = scale
        # Load image and mask names into an instance attribute
        file_name = os.path.join(self.data_dir, 'txt_files', f'{split}.txt')
        img_ids = [i_id.strip() for i_id in open(file_name) if '.npy' in i_id.strip()]
        self.files = []
        for name in img_ids:
            img_file = os.path.join(self.data_dir, 'imgs', name)
            mask_file = os.path.join(self.data_dir, 'masks', name)
            self.files.append({
                "img": img_file,
                "mask": mask_file
            })

    def resize(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image and mask. If scale=4 this is a 4x down-sample."""
        img_y, img_x = int(img.shape[1] / self.scale), int(img.shape[2] / self.scale)
        small_image = resize(img, (img.shape[0], img_y, img_x))
        mask_y, mask_x = int(mask.shape[0] / self.scale), int(mask.shape[1] / self.scale)
        small_mask = resize(mask, (mask_y, mask_x))
        return small_image, small_mask

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int) -> dict:
        """Dataset returns a dictionary with an image and mask key, each having an associated value of numpy array."""
        datafile = self.files[item]
        img = np.load(datafile["img"])
        mask = np.load(datafile["mask"])

        img, mask = self.resize(img, mask)

        assert img[0, :, :].shape == mask.shape, \
            f'Image and mask {item} should be the same size, but are {img.shape} and {mask.shape}'

        return {
            'image': torch.from_numpy(img),
            'mask': torch.from_numpy(mask)
        }
