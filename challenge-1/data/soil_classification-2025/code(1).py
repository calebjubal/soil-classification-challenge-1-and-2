import os
import shutil
import pandas as pd

def organize_train_images(base_directory="."):
    """
    Organizes images from the 'train' folder into subfolders based on 'soil_type'
    defined in 'train_labels.csv'. Handles various image extensions.

    Args:
        base_directory (str): The path to the directory containing 'train' folder
                              and 'train_labels.csv'.
    """
    train_folder_path = os.path.join(base_directory, "train")
    labels_csv_path = os.path.join(base_directory, "train_labels.csv")

    # --- 1. Validations ---
    if not os.path.exists(train_folder_path):
        print(f"Error: 'train' folder not found at {train_folder_path}")
        return
    if not os.path.isdir(train_folder_path):
        print(f"Error: '{train_folder_path}' is not a directory.")
        return
    if not os.path.exists(labels_csv_path):
        print(f"Error: 'train_labels.csv' not found at {labels_csv_path}")
        return

    print(f"Using train folder: {train_folder_path}")
    print(f"Using labels CSV: {labels_csv_path}")

    # --- 2. Read the CSV ---
    try:
        df_labels = pd.read_csv(labels_csv_path)
    except Exception as e:
        print(f"Error reading CSV file '{labels_csv_path}': {e}")
        return

    if 'image_id' not in df_labels.columns or 'soil_type' not in df_labels.columns:
        print("Error: CSV file must contain 'image_id' and 'soil_type' columns.")
        return

    print(f"Successfully read {len(df_labels)} entries from {os.path.basename(labels_csv_path)}.")

    # --- 3. Get unique soil types and create subfolders ---
    unique_soil_types = df_labels['soil_type'].unique()
    print(f"\nFound unique soil types: {', '.join(unique_soil_types)}")

    for soil_type in unique_soil_types:
        target_subfolder = os.path.join(train_folder_path, soil_type)
        
        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)
            print(f"Created folder: {target_subfolder}")
        else:
            if os.path.isdir(target_subfolder):
                 print(f"Folder already exists: {target_subfolder}")
            else:
                print(f"Error: A file exists with the name of a target subfolder: {target_subfolder}. Please remove it and retry.")
                return


    # --- 4. Pre-scan image files in the train directory (top level) ---
    print("\nScanning files in the top level of the train directory...")
    available_files_map = {} # Maps image_id (without extension) to full filename (with extension)
    
    # List only files, not subdirectories that might have been created in a previous run
    files_in_train_toplevel = [f for f in os.listdir(train_folder_path) 
                               if os.path.isfile(os.path.join(train_folder_path, f))]

    for filename_with_ext in files_in_train_toplevel:
        file_id_part, _ = os.path.splitext(filename_with_ext) # e.g., "img_ed005" from "img_ed005.jpeg"
        if file_id_part in available_files_map:
            print(f"Warning: Duplicate base image ID '{file_id_part}' found in train folder ('{available_files_map[file_id_part]}' and '{filename_with_ext}'). Using the latter.")
        available_files_map[file_id_part] = filename_with_ext # maps "img_ed005" to "img_ed005.jpeg"
    
    print(f"Found {len(available_files_map)} unique image files (by base ID) in {train_folder_path} (top level).")

    # --- 5. Move images to their respective folders ---
    moved_count = 0
    not_found_in_toplevel_count = 0
    already_in_subdir_count = 0
    error_count = 0

    print("\nMoving images...")
    for index, row in df_labels.iterrows():
        image_id_from_csv = row['image_id'] # e.g., "img_ed005" (no extension)
        soil_type = row['soil_type']
        
        # Attempt to find the actual filename with extension from the top-level scan
        actual_filename = available_files_map.get(image_id_from_csv)

        destination_folder_path = os.path.join(train_folder_path, soil_type) # e.g., train/Alluvial_soil

        if actual_filename: # File was found at the top level of 'train'
            source_image_path = os.path.join(train_folder_path, actual_filename)
            destination_image_path = os.path.join(destination_folder_path, actual_filename)

            # Ensure destination folder exists (it should have been created in step 3)
            if not os.path.exists(destination_folder_path):
                print(f"Error: Destination folder {destination_folder_path} unexpectedly does not exist. Skipping {actual_filename}.")
                error_count +=1
                continue
            if not os.path.isdir(destination_folder_path):
                print(f"Error: Destination path {destination_folder_path} is a file, not a folder. Skipping {actual_filename}.")
                error_count +=1
                continue

            try:
                shutil.move(source_image_path, destination_image_path)
                # print(f"Moved: {actual_filename} -> {destination_folder_path}")
                moved_count += 1
            except Exception as e:
                print(f"Error moving '{actual_filename}' to '{destination_folder_path}': {e}")
                error_count += 1
        else:
            # File not found at top-level. Check if it's already in the correct subdirectory.
            found_in_subdir = False
            if os.path.exists(destination_folder_path) and os.path.isdir(destination_folder_path):
                for f_in_sub in os.listdir(destination_folder_path):
                    # Check if f_in_sub is a file and its base name matches image_id_from_csv
                    if os.path.isfile(os.path.join(destination_folder_path, f_in_sub)):
                        base_f_in_sub, _ = os.path.splitext(f_in_sub)
                        if base_f_in_sub == image_id_from_csv:
                            found_in_subdir = True
                            break
            
            if found_in_subdir:
                # print(f"Image ID '{image_id_from_csv}' already in '{destination_folder_path}'. Skipping.")
                already_in_subdir_count +=1
            else:
                print(f"Warning: Image for ID '{image_id_from_csv}' not found at the top level of '{train_folder_path}' and not found in '{destination_folder_path}'. Skipping.")
                not_found_in_toplevel_count += 1
            
    print("\n--- Summary ---")
    print(f"Successfully moved {moved_count} images.")
    if not_found_in_toplevel_count > 0:
        print(f"{not_found_in_toplevel_count} images listed in CSV were not found at the top level of 'train' nor in their target subdirectories.")
    if already_in_subdir_count > 0:
        print(f"{already_in_subdir_count} images were already in their respective subdirectories.")
    if error_count > 0:
        print(f"Encountered {error_count} errors during processing.")
    print("Image organization complete.")

# --- How to run the script ---
if __name__ == "__main__":
    print("Starting image organization process...")

    # YOUR SPECIFIC PATH:
    your_data_path = r"C:\Users\Sarvesh\Desktop\New folder (4)\soil-classification-challenge-1-and-2\challenge-1\data\soil_classification-2025"
    
    print(f"Targeting base directory: {your_data_path}")
    organize_train_images(base_directory=your_data_path)

    print("Script finished.")