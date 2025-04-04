import os
import shutil

curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)

# Define paths
gray_folder = "../stimuli/gray"
mooney_folder = "../stimuli/mooney"
original_folder = "../stimuli/original"

mooney_source = "/Users/carlos/Documents/GitHub/things-mooney/stim/mooney"
original_source = "/Users/carlos/Downloads/object_images_CC0"

gray_suffix = '_gray'
mooney_suffix = '_mooney'

# Ensure the original_folder exists
os.makedirs(original_folder, exist_ok=True)

# Process gray images
for file_name in os.listdir(gray_folder):
    if file_name.endswith(f"{gray_suffix}.jpg"):
        object_label = file_name.replace(f"{gray_suffix}.jpg", "")
        original_file_name = f"{object_label}.jpg"
        original_file_path = os.path.join(original_source, original_file_name)
        if os.path.exists(original_file_path):
            shutil.copy(original_file_path, original_folder)

# Process mooney images
for file_name in os.listdir(gray_folder):
    if file_name.endswith(f"{gray_suffix}.jpg"):
        object_label = file_name.replace(f"{gray_suffix}.jpg", f"{mooney_suffix}.jpg")
        original_file_path = os.path.join(mooney_source, object_label)
        if os.path.exists(original_file_path):
            shutil.copy(original_file_path, mooney_folder)
