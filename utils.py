import torch
import torchvision
from dataset import UNetDataset
from torch.utils.data import DataLoader

def get_loaders(
    train_img_dir, 
    train_mask_dir, 
    val_img_dir, 
    val_mask_dir,
    batch_size, 
    train_transform, 
    val_transform, 
    num_workers,  # data loaded in parallel subprocesses
    pin_memory=True
):
    train_dataset = UNetDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_dataset = UNetDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def save_checkpoint():
    pass

def load_checkpoint():
    pass
 
def check_accuracy():
    pass

def save_predictions_as_imgs():
    pass
