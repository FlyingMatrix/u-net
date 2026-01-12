"""
    This study uses the ISBI-2012-Challenge | U-Net dataset, obtained from Kaggle: 
    https://www.kaggle.com/datasets/hamzamohiuddin/isbi-2012-challenge

    Data folder structure diagram:

    ├── data/
    │   ├── train/
    │   │   ├── imgs/frame_0001.png, ..., frame_0030.png
    │   │   └── labels/frame_0001.png, ..., frame_0030.png
    │   └── test/
    │       ├── imgs/frame_0001.png, ..., frame_0030.png
    │       └── labels/frame_0001.png, ..., frame_0030.png

"""

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class UNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # In our case, the image and its corresponding mask share the same name
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index]) 

        # Converts the image to grayscale ("L" = luminance, 8-bit pixels, values 0–255)
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask) # Albumentations transform
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
