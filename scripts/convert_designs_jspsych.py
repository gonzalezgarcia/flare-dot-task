import pandas as pd
import csv
import os
import json

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


#%% list stimuli in ../task/stimuli
## WATCH OUT FOR .DS_STORE FILES, IT WILL PREVENT JSPSYCH FROM LOADING THE STIMULI
stimuli = []

image_list = os.listdir('../task/stim')

# add "stim/" to the beginning of each image name
for i in range(len(image_list)):
    image_list[i] = 'stim/' + image_list[i]
    
with open("../task/preload_images.json", "w") as f:
    json.dump(image_list, f)

#%% do some image preprocessing to make sure all images are the same size
from PIL import Image
import os
import glob
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