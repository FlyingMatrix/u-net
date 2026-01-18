import torch
import torchvision
from dataset import UNetDataset
from torch.utils.data import DataLoader
import os

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

def save_checkpoint(state, filename="checkpoint.pth.tar"): 
    print(">>> Saving checkpoint...")
    torch.save(state, filename)
    """"
        state is a dictionary that contains everything needed to resume training later, for example:
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
    """

def load_checkpoint(checkpoint, model):
    print(">>> Loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])    

def save_predictions_as_images(dataloader, model, folder, device):
    model.eval()
    pred_dir = os.path.join(folder, "predictions")
    gt_dir = os.path.join(folder, "groundtruth_masks")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for idx, (imgs, masks) in enumerate(dataloader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(imgs))
            predictions = (probabilities > 0.5).float()  # (N, 1, H, W)
        # save each image in the batch
        for pred in range(predictions.size(0)):
            torchvision.utils.save_image(
                predictions[pred],
                os.path.join(pred_dir, f"{idx}_{pred}.png")
            )
            torchvision.utils.save_image(
                masks[pred].unsqueeze(0),
                os.path.join(gt_dir, f"{idx}_{pred}.png")
            )
