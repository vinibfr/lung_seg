from PIL import Image
import os

# Specify the path to the folder containing 4-channel images
input_folder_path = 'E:\data\Image'

# Specify the path to the folder where converted images will be saved
output_folder_path = 'E:/data/Image/Converted'


# Ensure the output folder exists, or create it if necessary
os.makedirs(output_folder_path, exist_ok=True)

# Iterate through each file in the input folder
for filename in os.listdir(input_folder_path):
    # Construct the full path to the input image
    input_image_path = os.path.join(input_folder_path, filename)

    # Load the 4-channel image (RGBA)
    image = Image.open(input_image_path)

    # Convert RGBA to RGB
    rgb_image = image.convert('RGB')

    # Construct the full path to the output image
    output_image_path = os.path.join(output_folder_path, filename)

    # Save the converted image to the output folder
    rgb_image.save(output_image_path)

print("Conversion completed.")
