import os
import shutil

def copy_files(source_folder, destination_folder):
    # Ensure that the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Walk through the source folder and its subfolders
    for foldername, subfolders, filenames in os.walk(source_folder):
        for filename in filenames:
            source_file = os.path.join(foldername, filename)
            destination_file = os.path.join(destination_folder, filename)

            # Copy the file
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {source_file} to {destination_file}")

# Example usage
source_folder = 'E:\preprocessing_lidc\data\Clean\Image'
destination_folder = 'E:\data\Clean\Image'
copy_files(source_folder, destination_folder)

source_folder = 'E:\preprocessing_lidc\data\Clean\Mask'
destination_folder = 'E:\data\Clean\Mask'
copy_files(source_folder, destination_folder)

source_folder = 'E:\preprocessing_lidc\data\Mask'
destination_folder = 'E:\data\Mask'
copy_files(source_folder, destination_folder)

source_folder = 'E:\preprocessing_lidc\data\Image'
destination_folder = 'E:\data\Image'
copy_files(source_folder, destination_folder)

source_folder = 'E:\preprocessing_lidc\data\Meta'
destination_folder = 'E:\data\Meta'
copy_files(source_folder, destination_folder)