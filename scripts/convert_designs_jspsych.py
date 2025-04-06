import pandas as pd
import csv
import os
import json
import shutil

curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)


#%% list all .csv files in the  designs folder
all_designs = []
for filename in os.listdir('../designs'):
    if filename.endswith('.csv'):
        df = pd.read_csv(f'../designs/{filename}')
        
        # convert to json and clean up (jspsych reads designs in json format)
        df_json = df.to_json(orient = "records", indent=0, lines = False)
        
        df_json = df_json.replace('{','{data: {')
        df_json = df_json.replace('}','}}')
        
        all_designs += [df_json]
        


#%% preprocess for json input to jspsych
# copy the resulting design_matrix into a json file with the format var DM = design_matrix
design_matrix = str(all_designs)
design_matrix = design_matrix.replace("'","")
design_matrix = design_matrix.replace("\\","")
# save the design matrix to a json file
with open("../task/design_matrices.json", "w") as f:
    f.write(f"var DM = {design_matrix}")


#%% copy dot images to the task stim folder
# get the names of the dot images
# Path to the source directory containing the folders with dot images

# Only copy .jpg files
source_dir = "../stimuli/output_dots"

# Path to the destination directory
destination_dir = "../task/stim"

# Iterate through all folders in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):  # Check if it's a folder
        # Copy all .jpg files from the folder to the destination directory
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith('.jpg'):  # Check if it's a .jpg file
                shutil.copy(file_path, destination_dir)
                
# now copy the images from stimuli/gray and stimuli/original that coincide with stimuli/image_files.txt
# read the image files. Copy them to destiantion_dir
image_files = []
with open("../stimuli/image_files.txt", "r") as f:
    for line in f:
        image_files.append(line.strip())
# copy the images to the destination directory
for image_file in image_files:
    # Check if the file exists in the source directory
    image_file = image_file.replace('_gray','_gray.jpg')
    source_path = os.path.join("../stimuli/gray", image_file)
    if os.path.isfile(source_path):
        # Copy the file to the destination directory
        shutil.copy(source_path, destination_dir)
    else:
        print(f"File {source_path} does not exist.")
# copy the images to the destination directory
for image_file in image_files:
    # remove the 'gray'
    image_file = image_file.replace('_gray','.jpg')
    # Check if the file exists in the source directory
    source_path = os.path.join("../stimuli/original", image_file)
    if os.path.isfile(source_path):
        # Copy the file to the destination directory
        shutil.copy(source_path, destination_dir)
    else:
        print(f"File {source_path} does not exist.")

#%% list stimuli in ../task/stimuli
## WATCH OUT FOR .DS_STORE FILES, IT WILL PREVENT JSPSYCH FROM LOADING THE STIMULI
stimuli = []

image_list = [file for file in os.listdir('../task/stim') if file.lower().endswith('.jpg')]

# add "stim/" to the beginning of each image name
for i in range(len(image_list)):
    image_list[i] = 'stim/' + image_list[i]
    
with open("../task/preload_images.json", "w") as f:
    json.dump(image_list, f)

#%% do some image preprocessing to make sure all images are the same size
from PIL import Image
import os
import glob
import shutil
# Set the directory containing the images
image_dir = "../task/stim/"
# Set the desired size
desired_size = (1200, 1200)
# Loop through all the images in the directory
for filename in glob.glob(os.path.join(image_dir, '*.jpg')):
    # Open the image
    img = Image.open(filename)
    # Resize the image
    img = img.resize(desired_size)
    # Save the image back to disk
    img.save(filename)