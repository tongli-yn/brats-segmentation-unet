# datasets/brats_dataset.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class BraTSDataset2D(Dataset):
    def __init__(self, meta_df, image_transform=None, label_transform=None):
        self.meta_df = meta_df
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        path = row['slice_path']

        with h5py.File(path, 'r') as f:
            image = f['image'][:]  # (H, W, 4)
            mask = f['mask'][:]    # (H, W, 3)

        image = (image - image.mean()) / (image.std() + 1e-5)
        image = np.transpose(image, (2, 0, 1))  # (4, H, W)
        mask = self.rgb_to_class(mask)[np.newaxis, ...]  # (1, H, W)

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            mask = self.label_transform(mask)

        return image.float(), mask.long().squeeze(0)

    def rgb_to_class(self, mask_rgb):
        colormap = {
            (0, 0, 0): 0,
            (255, 0, 0): 1,
            (0, 255, 0): 2,
            (0, 0, 255): 3,
            (255, 255, 0): 4
        }
        h, w, _ = mask_rgb.shape
        mask_flat = mask_rgb.reshape(-1, 3)
        mask_class = np.zeros((h * w,), dtype=np.uint8)
        for rgb, cls in colormap.items():
            mask_class[np.all(mask_flat == rgb, axis=1)] = cls
        return mask_class.reshape(h, w)

