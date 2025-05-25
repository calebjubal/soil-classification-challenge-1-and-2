"""
Author: Annam.ai IIT Ropar
Team Name: SoilClassifiers
Team Members: Member-1, Member-2, Member-3, Member-4, Member-5
Leaderboard Rank: 120
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Configuration from training (relevant for data loading)
IMG_SIZE = 224
# BATCH_SIZE_TRAIN = 32 # BATCH_SIZE might be different for training vs inference preprocessing steps

# Device configuration (useful if any preprocessing steps are on GPU, though usually CPU is fine for loading)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset (same as in training and inference)
class SoilDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.image_files = [f for f in os.listdir(folder)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name # Return image_name for potential tracking

# Transforms (typically for training data)
def get_train_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Add any training-specific augmentations here if needed in the future
        # e.g., transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Transforms (typically for validation/test data - often same as train but without augmentation)
def get_val_test_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_train_dataloader(train_folder_path, batch_size, img_size=IMG_SIZE, num_workers=2):
    """
    Creates and returns a DataLoader for the training dataset.
    """
    train_transforms_pipeline = get_train_transforms(img_size)
    train_dataset = SoilDataset(
        folder=train_folder_path,
        transform=train_transforms_pipeline
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    return train_loader

def get_test_dataloader(test_folder_path, batch_size, img_size=IMG_SIZE, num_workers=2):
    """
    Creates and returns a DataLoader for the test dataset.
    Used during inference/postprocessing step.
    """
    test_transforms_pipeline = get_val_test_transforms(img_size)
    test_dataset = SoilDataset(
        folder=test_folder_path,
        transform=test_transforms_pipeline
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # No shuffle for test/val
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    return test_loader


def preprocessing_example():
    print("Demonstrating preprocessing pipeline...")
    # Example usage:
    # Note: You'll need to have a dummy folder structure for this to run without error,
    # or point it to your actual data paths.
    # For example:
    # dummy_train_folder = "dummy_data/train"
    # os.makedirs(dummy_train_folder, exist_ok=True)
    # # Create a dummy image
    # try:
    #     Image.new('RGB', (100, 100), color = 'red').save(os.path.join(dummy_train_folder, "dummy.jpg"))
    # except ImportError:
    #     print("Pillow (PIL) is not installed. Cannot create dummy image.")
    #     return

    # For Kaggle paths, it would be:
    train_folder = "/kaggle/input/soil-classification-part-2/soil_competition-2025/train"
    
    # Check if the path exists to avoid errors if not on Kaggle
    if not os.path.exists(train_folder):
        print(f"Warning: Training data folder not found at {train_folder}. Preprocessing demo will be limited.")
        # Create a dummy path for demonstration purposes if the original path doesn't exist
        train_folder = "temp_train_data"
        os.makedirs(train_folder, exist_ok=True)
        try:
            Image.new('RGB', (60, 60), color = 'blue').save(os.path.join(train_folder, "example.jpg"))
            print(f"Created dummy data in {train_folder} for demonstration.")
        except Exception as e:
            print(f"Could not create dummy image: {e}")
            return


    print(f"Using image size: {IMG_SIZE}")
    train_loader = get_train_dataloader(train_folder_path=train_folder, batch_size=4, img_size=IMG_SIZE) # small batch for demo

    print(f"Number of training samples: {len(train_loader.dataset)}")
    
    if len(train_loader.dataset) > 0:
        print("Fetching a batch from train_loader...")
        try:
            imgs, img_names = next(iter(train_loader))
            print(f"Batch of images shape: {imgs.shape}") # Expected: [batch_size, 3, IMG_SIZE, IMG_SIZE]
            print(f"First image name in batch: {img_names[0]}")
            print(f"Image tensor min/max: {imgs.min().item():.2f}/{imgs.max().item():.2f} (after normalization)")
        except StopIteration:
            print("Train loader is empty. Cannot fetch a batch.")
        except RuntimeError as e:
            print(f"Error fetching batch (ensure data folder '{train_folder}' is not empty and accessible): {e}")

    else:
        print("Train dataset is empty. Cannot demonstrate batch fetching.")

    print("Preprocessing setup complete.")
    return 0

if __name__ == '__main__':
    preprocessing_example()
