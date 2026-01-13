import torch
import albumentations as A # !pip install albumentations opencv-python
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True # page-locked memory, which allows faster and asynchronous transfers from CPU to GPU
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/imgs"
TRAIN_MASK_DIR = "data/train/labels"
VAL_IMG_DIR = "data/test/imgs"
VAL_IMG_DIR = "data/test/imgs"

