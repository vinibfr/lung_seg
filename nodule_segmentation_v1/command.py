import subprocess

# Define the command to be executed
command = "python train.py --name UNET --augmentation True"

# Run the command using subprocess
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

command = "python train.py --name NestedUNET --augmentation True"

# Run the command using subprocess
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

# Define the command to be executed
command = "python train.py --name UNET --augmentation False"

# Run the command using subprocess
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

command = "python train.py --name NestedUNET --augmentation False"

# Run the command using subprocess
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")    