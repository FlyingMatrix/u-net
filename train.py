import torch
import albumentations as A # !pip install albumentations opencv-python
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import load_checkpoint, save_checkpoint, get_loaders

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True # Page-locked memory, which allows faster and asynchronous transfers from CPU to GPU
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/imgs"
TRAIN_MASK_DIR = "data/train/labels"
VAL_IMG_DIR = "data/test/imgs"
VAL_MASK_DIR = "data/test/labels"

# Data preparation
def data_prep():
    train_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        # Single-channel normalization: mean=(0.5,), std=(0.5,) is used to zero-center grayscale images 
        # ... and scale them to roughly [-1, 1], which improves neural network training stability.
        A.Normalize(                
            mean=(0.5,),
            std=(0.5,),
            max_pixel_value=255.0
        ), 
        ToTensorV2() # HWC -> CHW, numpy -> torch.Tensor, output dtype: float32
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=(0.5,),
            std=(0.5,),
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    return train_loader, val_loader

# Train loop
def train():
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() 
    """
        nn.BCEWithLogitsLoss = sigmoid + BCE in a single, numerically stable function
        Input = logits instead of probabilities, widely used in binary classification and binary segmentation
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")
    """
        GradScaler helps prevent underflow (gradients can become 0) when training with mixed precision on GPUs by:
        - Scaling the loss before backpropagation
        - Unscaling gradients before the optimizer step
        - Skipping optimizer steps if gradients contain NaNs/Infs

        Mixed precision means using more than one numeric precision during training. Usually:
        - FP32 (float32) -> for stability
        - FP16 (float16) or BF16 -> for speed & memory
    """

    train_loader, val_loader = data_prep()

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint.pth.tar"), model)

    for epoch in range(NUM_EPOCHS):
        # train the model
        model.train()
        train_loader = tqdm(train_loader)
        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE) # img.shape => (N, C=1, H, W)
            masks = masks.float().unsqueeze(1).to(DEVICE) # mask.shape -> (N, H, W) -> (N, 1, H, W)
            # forward
            with torch.amp.autocast(device_type="cuda"):
                predictions = model(imgs)
                loss = loss_fn(predictions, masks)
            # backward
            optimizer.zero_grad()               # clear old gradients
            scaler.scale(loss).backward()       # scale the loss, then backpropagate
            scaler.step(optimizer)              # unscale gradients and update weights (safely)
            scaler.update()                     # adjust the scaling factor dynamically
            # update tqdm progress bar with current training loss
            train_loader.set_postfix(loss=loss.item())

        # save the model
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        # validation
        model.eval()




if __name__ == "__main__":
    train()