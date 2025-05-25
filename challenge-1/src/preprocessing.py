"""
Author: Annam.ai IIT Ropar
Team Name: SoilClassifiers
Team Members: Caleb Chandrasekar, Sarvesh Chandran, Swaraj Bhattacharjee, Karan Singh, Saatvik Tyagi
Leaderboard Rank: 103
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

# --- Configuration & Constants ---
# (These can be imported and used by both training and inference scripts)

# Device configuration (can be set by the script that imports this module)
# Defaulting to CPU, but can be overridden.
# Example in training/inference:
# import preprocessing
# preprocessing.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Image and Class definitions
IMG_SIZE = 384  # As used in both notebooks
CLASSES = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


# --- Transformations ---

def get_base_transform(img_size=IMG_SIZE):
    """
    Returns the base transformation pipeline (resize, to_tensor, normalize).
    Used for training, validation, and as a base for TTA.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_tta_transforms(img_size=IMG_SIZE):
    """
    Returns a list of transformation pipelines for Test-Time Augmentation (TTA).
    """
    base_transform = get_base_transform(img_size) # To ensure consistency if IMG_SIZE changes

    tta_transforms_list = [
        base_transform,  # Original transform
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0), # Always flip
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # Add more TTA transforms if desired
        # e.g., RandomRotation, CenterCrop after a slightly larger Resize
        transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))), # Resize slightly larger
            transforms.CenterCrop(img_size), # Then center crop
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    ]
    return tta_transforms_list


# --- Dataset Definitions ---

class SoilLabelledDataset(Dataset):
    """
    Dataset for soil classification with labels (e.g., for training and validation).
    Reads image paths and labels from a CSV file.
    """
    def __init__(self, csv_file, root_dir, transform=None, class_to_idx_map=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx_map if class_to_idx_map is not None else CLASS_TO_IDX

        # Pre-validate image paths to some extent or handle errors in __getitem__
        # For example, ensure image_id column exists
        if 'image_id' not in self.annotations.columns or \
           (len(self.annotations.columns) > 1 and 'soil_type' not in self.annotations.columns): # second col for label
            raise ValueError("CSV file must contain 'image_id' and 'soil_type' columns.")


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id_original = self.annotations.iloc[idx, 0] # First column is image_id
        image_id = str(image_id_original)

        # Standardize image extension handling (as in training.ipynb)
        if '.' not in image_id:
            image_id += '.jpg' # Default to .jpg if no extension
        else:
            base, ext = os.path.splitext(image_id)
            image_id = base + ext.lower() # Ensure lowercase extension

        img_path = os.path.join(self.root_dir, image_id)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Image not found at {img_path} (original ID: {image_id_original}). Please check path and CSV.")
            # Return a placeholder or raise an error
            # For robustness, one might return a dummy tensor and a special label,
            # but this often complicates batch collation. Better to ensure data integrity.
            raise FileNotFoundError(f"Image not found: {img_path}")


        if self.transform:
            image = self.transform(image)

        label_str = self.annotations.iloc[idx, 1].strip() # Second column is soil_type
        label = self.class_to_idx[label_str]

        return image, label

class SoilTestDataset(Dataset):
    """
    Dataset for soil classification test images (image_id only).
    Reads image IDs from a CSV file or a list.
    """
    def __init__(self, image_id_source, root_dir, transform=None):
        """
        image_id_source: Path to CSV file with 'image_id' column, or a list of image_ids.
        root_dir: Directory containing the test images.
        transform: The transformation to apply to each image.
        """
        if isinstance(image_id_source, str): # Path to CSV
            self.image_ids_df = pd.read_csv(image_id_source)
            if 'image_id' not in self.image_ids_df.columns:
                raise ValueError("Test CSV file must contain 'image_id' column.")
            self.image_ids = self.image_ids_df['image_id'].tolist()
        elif isinstance(image_id_source, list):
            self.image_ids = image_id_source
        else:
            raise TypeError("image_id_source must be a CSV path or a list of image IDs.")

        self.root_dir = root_dir
        self.transform = transform # This would typically be the base_transform for inference

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id_original = self.image_ids[idx]
        image_id_processed = str(image_id_original)

        # Standardize image extension handling (as in inference.ipynb)
        if '.' not in image_id_processed:
            image_id_processed += '.jpg'
        else:
            base, ext = os.path.splitext(image_id_processed)
            image_id_processed = base + ext.lower()

        img_path = os.path.join(self.root_dir, image_id_processed)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Test image not found at {img_path} (original ID: {image_id_original}).")
            raise FileNotFoundError(f"Test image not found: {img_path}")


        if self.transform:
            image = self.transform(image)

        return image, image_id_original # Return original ID for submission mapping


# --- DataLoader Helper Functions ---

def get_train_val_dataloaders(csv_path,
                              root_dir,
                              batch_size,
                              img_size=IMG_SIZE,
                              val_split=0.2,
                              num_workers=2,
                              pin_memory=True,
                              seed=42):
    """
    Creates training and validation DataLoaders from a labelled dataset.
    """
    transform = get_base_transform(img_size)
    dataset = SoilLabelledDataset(
        csv_file=csv_path,
        root_dir=root_dir,
        transform=transform,
        class_to_idx_map=CLASS_TO_IDX
    )

    if val_split <= 0 or val_split >= 1:
        # Use full dataset for training if val_split is not sensible
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = None # No validation loader
        print(f"Using full dataset for training. Train size: {len(dataset)}")
        return train_loader, val_loader

    dataset_size = len(dataset)
    val_count = int(dataset_size * val_split)
    train_count = dataset_size - val_count

    # Ensure reproducible splits
    generator = torch.Generator().manual_seed(seed)
    train_data, val_data = torch.utils.data.random_split(dataset, [train_count, val_count], generator=generator)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")
    return train_loader, val_loader


def get_test_dataloader(image_id_source,
                        root_dir,
                        batch_size,
                        img_size=IMG_SIZE,
                        num_workers=2,
                        pin_memory=True):
    """
    Creates a DataLoader for test images.
    Note: For TTA, inference is often done image by image, applying multiple transforms.
    This DataLoader provides images with only the base transform.
    The TTA loop would then apply further augmentations.
    """
    transform = get_base_transform(img_size)
    test_dataset = SoilTestDataset(
        image_id_source=image_id_source,
        root_dir=root_dir,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print(f"Test dataset size: {len(test_dataset)}")
    return test_loader

# --- Helper function to load a single image ---
# (Useful for the TTA loop in inference if not using a DataLoader for TTA)

def load_and_transform_image(image_path, transform_pipeline):
    """
    Loads a single image from path and applies a given transformation pipeline.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return transform_pipeline(image)
    except FileNotFoundError:
        print(f"ERROR: Image not found at {image_path}.")
        raise
    except Exception as e:
        print(f"ERROR: Could not load or transform image {image_path}: {e}")
        raise


# --- Example Usage ---
def preprocessing_example():
    print("--- Preprocessing Example ---")
    # This example assumes dummy data or correct paths for Kaggle.
    # Create dummy files and folders for local testing if needed.

    # Common paths (replace with actual Kaggle paths or local test paths)
    dummy_train_csv_path = "dummy_train_labels.csv"
    dummy_train_dir = "dummy_train_images"
    dummy_test_ids_csv_path = "dummy_test_ids.csv"
    dummy_test_dir = "dummy_test_images"

    # Create dummy data for local testing
    os.makedirs(dummy_train_dir, exist_ok=True)
    os.makedirs(dummy_test_dir, exist_ok=True)

    if not os.path.exists(dummy_train_csv_path):
        dummy_train_data = {'image_id': [], 'soil_type': []}
        for i in range(5):
            img_name = f"train_img_{i}.jpg"
            try:
                Image.new('RGB', (60, 60), color = 'red').save(os.path.join(dummy_train_dir, img_name))
                dummy_train_data['image_id'].append(img_name)
                dummy_train_data['soil_type'].append(CLASSES[i % len(CLASSES)])
            except Exception as e:
                print(f"Could not create dummy train image {img_name}: {e}")
        pd.DataFrame(dummy_train_data).to_csv(dummy_train_csv_path, index=False)
        print(f"Created dummy train CSV: {dummy_train_csv_path} and images in {dummy_train_dir}")

    if not os.path.exists(dummy_test_ids_csv_path):
        dummy_test_data = {'image_id': []}
        for i in range(3):
            img_name = f"test_img_{i}.png" # Test with different extension
            try:
                Image.new('RGB', (50, 50), color = 'blue').save(os.path.join(dummy_test_dir, img_name))
                dummy_test_data['image_id'].append(img_name)
            except Exception as e:
                print(f"Could not create dummy test image {img_name}: {e}")
        pd.DataFrame(dummy_test_data).to_csv(dummy_test_ids_csv_path, index=False)
        print(f"Created dummy test IDs CSV: {dummy_test_ids_csv_path} and images in {dummy_test_dir}")

    print(f"\nDevice being used by preprocessing module (default): {device}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Classes: {CLASSES}")

    # 1. Get transformations
    train_transform = get_base_transform()
    tta_list = get_tta_transforms()
    print(f"\nBase transform: {train_transform}")
    print(f"Number of TTA transforms: {len(tta_list)}")

    # 2. Get DataLoaders
    print("\nTrying to create DataLoaders with dummy data...")
    try:
        train_loader, val_loader = get_train_val_dataloaders(
            csv_path=dummy_train_csv_path,
            root_dir=dummy_train_dir,
            batch_size=2,
            img_size=IMG_SIZE, # Use global IMG_SIZE
            val_split=0.4 # 2 for val, 3 for train
        )
        if train_loader:
            print(f"Train loader created. Number of batches: {len(train_loader)}")
            img_batch, label_batch = next(iter(train_loader))
            print(f"Sample train batch - images shape: {img_batch.shape}, labels: {label_batch}")
        if val_loader:
            print(f"Validation loader created. Number of batches: {len(val_loader)}")

        test_loader = get_test_dataloader(
            image_id_source=dummy_test_ids_csv_path,
            root_dir=dummy_test_dir,
            batch_size=1,
            img_size=IMG_SIZE
        )
        if test_loader:
            print(f"Test loader created. Number of batches: {len(test_loader)}")
            img_batch, id_batch = next(iter(test_loader))
            print(f"Sample test batch - images shape: {img_batch.shape}, ids: {id_batch}")

    except Exception as e:
        print(f"Error during DataLoader creation or iteration: {e}")
        print("Ensure dummy data (CSVs and image files) exists and paths are correct.")

    # 3. Example of loading a single image for TTA (manual loop style)
    print("\nExample of loading single image for TTA:")
    if os.path.exists(dummy_test_dir) and len(os.listdir(dummy_test_dir)) > 0:
        sample_test_img_name = os.listdir(dummy_test_dir)[0]
        sample_test_img_path = os.path.join(dummy_test_dir, sample_test_img_name)
        print(f"Loading image: {sample_test_img_path} with TTA transforms")
        
        original_image_pil = Image.open(sample_test_img_path).convert('RGB')
        
        tta_outputs = []
        for i, tta_transform_pipeline in enumerate(tta_list):
            try:
                # Option 1: Using the helper function
                # transformed_tensor = load_and_transform_image(sample_test_img_path, tta_transform_pipeline)
                
                # Option 2: Applying transform to already loaded PIL image
                transformed_tensor = tta_transform_pipeline(original_image_pil)
                
                tta_outputs.append(transformed_tensor)
                print(f"  TTA Transform {i} output shape: {transformed_tensor.shape}")
            except Exception as e:
                print(f"  Error applying TTA transform {i} to {sample_test_img_path}: {e}")
        if tta_outputs:
            # Example: Stack TTA results (if model expects batch dim)
            stacked_tta_tensors = torch.stack(tta_outputs)
            print(f"Stacked TTA tensors shape: {stacked_tta_tensors.shape}")
    else:
        print("Skipping single image TTA example as dummy test directory is empty or not found.")

    print("\n--- End of Preprocessing Example ---")
    return 0

if __name__ == '__main__':
    # To make this runnable, set the device (optional, defaults to CPU)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessing_example()