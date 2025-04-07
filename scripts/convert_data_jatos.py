import pandas as pd
import numpy as np
import csv
import os

# %% Config
curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)

#%% list all .txt files in the  designs folder
all_raw = []
for filename in os.listdir('../results'):
    if filename.endswith('.txt'):
        all_raw += [filename]
        print(filename)
        
    

filename = os.path.join('../results', all_raw[0])
# Step 1: Read the data from the file into a list of strings
with open(filename, 'r') as f:
    lines = f.readlines()

# Step 2: Find the indices of the header lines
header_indices = [i for i, line in enumerate(
    lines) if line.startswith('"rt"') or line.startswith('"success"')]

# Step 3: Split the data into chunks based on the headers
chunks = []
for i, index in enumerate(header_indices):
    if i == len(header_indices) - 1:
        chunk = lines[index:]
    else:
        chunk = lines[index:header_indices[i+1]]
    chunks.append(chunk)

# Step 4: # Process each chunk into a DataFrame
dfs = []

for chunk in chunks:
    # Create a CSV reader for the chunk
    reader = csv.reader(chunk)
    # Get the headers for the chunk
    headers = next(reader)
    # Read the chunk into a DataFrame
    df = pd.DataFrame.from_records(reader, columns=headers)
    if len(df) > 5: # if more than 20 trials...
        df.dropna(inplace=True)
        # reset index to have a fully increasing one
        df.reset_index(drop=True, inplace=True)
        
        # reoder columns
        df = df[['subject', 'subject_ProlificID', 'designmatrix_ID','age', 'gender', 'handedness', 'nationality',
                    'trial_number', 'block', 'block_type','image_id','filename',
                    'image_type', 'manipulation','phase','dot_position',
                    'RawResponse','dot_resp','dot_acc','rt', 'verbal_id']]

        # convert columns to numeric
        cols = ['age', 'trial_number', 'block', 'rt', 'dot_acc']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
        # Add the DataFrame to the list
        dfs.append(df)
        
# Merge all the DataFrames
data = pd.concat(dfs, ignore_index=True)