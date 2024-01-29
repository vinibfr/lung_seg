import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def numpy_arrays_to_pil_and_save(input_folders, output_folders):
    """
    Convert NumPy arrays to PIL Images and save them in corresponding output folders.

    Parameters:
    - input_folders: List of paths to folders containing NumPy arrays.
    - output_folders: List of paths to folders where PIL Images will be saved.
    """
    # Ensure the output folders exist
    for folder in output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for input_folder, output_folder in zip(input_folders, output_folders):
        # List all files in the input folder
        file_list = os.listdir(input_folder)

        for file_name in file_list:
            # Assuming the files are NumPy arrays with '.npy' extension
            if file_name.endswith('.npy'):
                file_path = os.path.join(input_folder, file_name)

                # Load the NumPy array
                array_data = np.load(file_path)
                
                # Save the PIL Image in the output folder with the same filename
                output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.png')
                plt.imsave(output_path, array_data)                
                print(f"Copied: {input_folder} to {output_folder} file: {file_name}")

# Example usage:
# Replace 'input_folder_paths' and 'output_folder_paths' with your actual folder paths
#input_folder_paths = ['E:\lung_nodule\preprocessing\data\Image','E:\lung_nodule\preprocessing\data\Mask','E:\lung_nodule\preprocessing\data\Clean\Image','E:\lung_nodule\preprocessing\data\Clean\Mask']
#output_folder_paths = ['E:\data\Image','E:\data\Mask','E:\data\Clean\Image','E:\data\Clean\Mask']


input_folder_paths = ['E:\lung_nodule\preprocessing\data\Mask']
output_folder_paths = ['E:\data\Mask']


numpy_arrays_to_pil_and_save(input_folder_paths, output_folder_paths)