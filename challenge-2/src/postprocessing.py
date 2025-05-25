"""
Author: Annam.ai IIT Ropar
Team Name: SoilClassifiers
Team Members: Member-1, Member-2, Member-3, Member-4, Member-5
Leaderboard Rank: <Your Rank>
"""

import os
import torch
import torch.nn as nn
# Re-using SoilDataset and get_test_dataloader from preprocessing.py
# If preprocessing.py is in the same directory, you can import:
# from preprocessing import SoilDataset, get_test_dataloader, IMG_SIZE
# For self-contained script, we redefine/copy them:

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration (should match training and inference settings)
IMG_SIZE = 224  # Must be consistent
INFERENCE_BATCH_SIZE = 1 # Usually 1 for inference if saving individual results or per-image logic

# Device configuration
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # From inference notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Copied/Adapted from preprocessing.py or inference.ipynb for self-containment ---
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
        return img, img_name # Return image_name

def get_test_transforms(img_size=IMG_SIZE): # Renamed for clarity if used alongside train transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], # Standard ImageNet stats
                             [0.229, 0.224, 0.225])
    ])

def get_inference_dataloader(test_folder_path, batch_size, img_size=IMG_SIZE, num_workers=2):
    test_transforms_pipeline = get_test_transforms(img_size)
    test_dataset = SoilDataset(
        folder=test_folder_path,
        transform=test_transforms_pipeline
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    return test_loader
# --- End of Copied/Adapted section ---


# Autoencoder Model Definition (must match training.py and autoencoder.pth)
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,   3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Visualization helper (from inference.ipynb)
def show_reconstruction(orig, recon, title_suffix=""):
    """Shows original and reconstructed images. Input tensors are expected to be normalized."""
    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(orig.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(orig.device)

    orig_display = orig.cpu().squeeze().clone() * std.cpu() + mean.cpu()
    orig_display = orig_display.permute(1, 2, 0).numpy().clip(0, 1)

    recon_display = recon.cpu().squeeze().clone() * std.cpu() + mean.cpu() # Note: Sigmoid output is 0-1, then it's normalized for loss.
                                                                       # If loss is against normalized, then recon is also "normalized-like"
                                                                       # If model output sigmoid is directly compared to 0-1 original (before norm),
                                                                       # then this denorm step for recon might be different.
                                                                       # Given loss = nn.MSELoss()(recon, imgs), imgs is normalized, so recon is aiming for normalized target.
    recon_display = recon_display.permute(1, 2, 0).numpy().clip(0, 1)


    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_display)
    axes[0].set_title(f"Original {title_suffix}")
    axes[0].axis('off')
    axes[1].imshow(recon_display)
    axes[1].set_title(f"Reconstruction {title_suffix}")
    axes[1].axis('off')
    plt.show()

def run_inference_and_postprocess(test_data_folder, model_path, threshold_path, output_csv_path="submission.csv", visualize_count=0):
    """
    Loads the model, runs inference on test data, applies threshold,
    visualizes some reconstructions, and saves the submission file.
    """
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    model = Autoencoder().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot proceed.")
        return 1
    except Exception as e:
        print(f"Error loading model: {e}. Cannot proceed.")
        return 1
    model.eval()

    print(f"Loading threshold from: {threshold_path}")
    try:
        threshold = float(np.load(threshold_path))
    except FileNotFoundError:
        print(f"Error: Threshold file not found at {threshold_path}. Cannot proceed.")
        return 1
    except Exception as e:
        print(f"Error loading threshold: {e}. Cannot proceed.")
        return 1
    print(f"Loaded threshold: {threshold:.6f}")

    print(f"Preparing test data loader from: {test_data_folder}")
    if not os.path.exists(test_data_folder) or not os.listdir(test_data_folder):
        print(f"Warning: Test data folder '{test_data_folder}' is empty or does not exist. Creating dummy data for demonstration.")
        # Create dummy data if test folder is empty or not found
        os.makedirs(test_data_folder, exist_ok=True)
        try:
            Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = 'green').save(os.path.join(test_data_folder, "test_dummy_1.jpg"))
            Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = 'yellow').save(os.path.join(test_data_folder, "test_dummy_2.jpg"))
            print(f"Created dummy test images in {test_data_folder}")
        except Exception as e:
            print(f"Could not create dummy test images: {e}")


    test_loader = get_inference_dataloader(test_data_folder, INFERENCE_BATCH_SIZE, IMG_SIZE)
    
    if len(test_loader.dataset) == 0:
        print("Test dataset is empty. Cannot perform inference.")
        return 1

    results = []
    criterion = nn.MSELoss() # To calculate reconstruction loss

    print("Starting inference loop...")
    with torch.no_grad():
        for i, (imgs, ids) in enumerate(test_loader):
            imgs = imgs.to(device)
            recon = model(imgs)
            loss = criterion(recon, imgs).item() # Per-image loss as batch_size=1
            label = 1 if loss < threshold else 0 # 1 for "normal" (low error), 0 for "anomaly" (high error)
            
            # ids is a tuple of image names, imgs is a batch.
            # If batch_size is 1, ids[0] is the image name.
            current_id = ids[0] if isinstance(ids, tuple) or isinstance(ids, list) else ids
            results.append((current_id, label))

            if i < visualize_count:
                print(f"Visualizing reconstruction for: {current_id} (Loss: {loss:.4f}, Label: {label})")
                show_reconstruction(imgs[0], recon[0], title_suffix=f"- {current_id[:15]}")

    print(f"Inference complete. Processed {len(results)} images.")

    # Save submission
    submission_df = pd.DataFrame(results, columns=["image_id", "label"])
    submission_df.to_csv(output_csv_path, index=False)
    print(f"✅ Submission file saved to: {output_csv_path}")

    return 0

def postprocessing():
    print("This is the main function for postprocessing tasks.")
    
    # Define paths (these would typically be arguments or configured)
    # For Kaggle:
    test_dir     = "/kaggle/input/soil-classification-part-2/soil_competition-2025/test"
    model_file   = "autoencoder.pth" # Assuming it's in the current working directory or a specified path
    thresh_file  = "threshold.npy"   # Assuming it's in the current working directory or a specified path
    
    # Fallback for local testing if Kaggle paths don't exist
    if not os.path.exists(test_dir):
        print(f"Kaggle test directory '{test_dir}' not found. Using local 'dummy_test_data'.")
        test_dir = "dummy_test_data" # Create this folder locally for testing
        os.makedirs(test_dir, exist_ok=True)
        # Create dummy model and threshold if they don't exist for local run
        if not os.path.exists(model_file):
            print(f"'{model_file}' not found. This script expects a trained model.")
            # As a placeholder, we can't really run without a model.
            # Consider adding a dummy model creation for pure structural testing if needed.
        if not os.path.exists(thresh_file):
            print(f"'{thresh_file}' not found. Creating a dummy threshold.npy.")
            np.save(thresh_file, np.array(0.1))


    # Run the full inference and postprocessing pipeline
    # Set visualize_count > 0 to see some image reconstructions
    status = run_inference_and_postprocess(
        test_data_folder=test_dir,
        model_path=model_file,
        threshold_path=thresh_file,
        output_csv_path="submission.csv",
        visualize_count=2 # Show first 2 reconstructions
    )
    
    if status == 0:
        print("Postprocessing script finished successfully.")
    else:
        print("Postprocessing script encountered errors.")
    return status

if __name__ == '__main__':
    # This will run when the script is executed directly
    # Ensure you have:
    # 1. A 'dummy_test_data' folder with some .jpg/.png images (if not on Kaggle)
    # 2. 'autoencoder.pth' (your trained model)
    # 3. 'threshold.npy' (your calculated threshold)
    #    in the same directory as this script, or update paths accordingly.
    
    # For the script to run without errors locally (without actual model/threshold files),
    # the run_inference_and_postprocess function has some checks.
    # However, actual inference needs the model.
    
    # Example: Create dummy model and threshold for structural testing
    # This is NOT a functional model, just to allow the script to run through
    if not os.path.exists("autoencoder.pth"):
        print("Creating a dummy autoencoder.pth for testing structure. THIS IS NOT A TRAINED MODEL.")
        dummy_model = Autoencoder()
        torch.save(dummy_model.state_dict(), "autoencoder.pth")
    if not os.path.exists("threshold.npy"):
        print("Creating a dummy threshold.npy for testing structure.")
        np.save("threshold.npy", np.array(0.01)) # Dummy threshold

    postprocessing()