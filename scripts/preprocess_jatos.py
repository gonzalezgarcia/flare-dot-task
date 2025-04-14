"""
Preprocess JATOS .txt file with multiple participants into individual CSVs.
"""

import os
import csv
import pandas as pd
from pathlib import Path
import logging
import argparse
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
import numpy as np
import re


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# File paths
exp_code = 'exp_3'
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / exp_code/ "raw"
DERIV_DIR = BASE_DIR / "data" / exp_code / "derivatives"
DERIV_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = [
    'subject', 'subject_ProlificID', 'designmatrix_ID', 'age', 'gender', 'handedness', 'nationality',
    'trial_number', 'block', 'block_type', 'image_id', 'filename',
    'image_type', 'manipulation', 'phase', 'dot_position',
    'RawResponse', 'dot_resp', 'dot_acc', 'rt', 'verbal_id'
]

def split_into_chunks(lines):
    header_indices = [
        i for i, line in enumerate(lines)
        if line.startswith('"rt"') or line.startswith('"success"')
    ]
    chunks = []
    for i, start_idx in enumerate(header_indices):
        end_idx = header_indices[i + 1] if i < len(header_indices) - 1 else len(lines)
        chunks.append(lines[start_idx:end_idx])
    return chunks

def process_chunk(chunk):
    try:
        reader = csv.reader(chunk)
        headers = next(reader)
        if len(headers) < 10:
            raise ValueError("Too few columns to be a valid participant chunk")

        df = pd.DataFrame.from_records(reader, columns=headers)

        if df.empty:
            raise ValueError("Empty dataframe")
        if 'block_type' not in df.columns:
            raise ValueError("Missing 'block_type' column")

        df = df[df['block_type'].isin(['dot', 'recognition'])]

        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            return None, f"missing columns: {missing}"
        return df, None

    except Exception as e:
        return None, f"Exception: {str(e)}"
    


## define a function to calculate semantic distance between given and true label
# load embeddings from THINGS and compute distance matrix
# in this matrix, each row is a concept (in the same order as uniqueID, see below)
# and the columns represent the distances to different concepts
UTILS_DIR = BASE_DIR / "scripts" / "utils"

embeddings = pd.read_csv(UTILS_DIR / 'sensevec_augmented_with_wordvec.csv', header=None)
distances = pdist(embeddings, 'cosine')
dist_matrix = squareform(distances)

concepts = pd.read_csv(UTILS_DIR / 'things_concepts.csv')
synonyms_distance = list(concepts["WordNet Synonyms"])
uniqueID = list(concepts["uniqueID"])

# substitute "_" for spaces in synonyms
# for i in range(len(synonyms_distance)):
#     words = synonyms_distance[i].split("_") # Split the row into individual words using the split() function
#     synonyms_distance[i] = " ".join(words) # Join the words together with spaces using the join() function

# add actual filename (from uniqueID) to each row of synonyms
for i in range(len(synonyms_distance)):
    synonyms_distance[i] = synonyms_distance[i] + ", " + uniqueID[i]
    
def compute_sd(data):
    # Replace spaces with underscore (to be able to compare with uniqueID)
    data['image_id'] = data['image_id'].str.replace(' ', '_')
    
    # fix a few polysemic labels
    def fix_polysemia(raw_data):
        # replace string in "verbal_id" column with correct string
        data = raw_data.copy()

        #data["verbal_id"] = data["verbal_id"].replace("bottle","bottle.n.01")
        data["verbal_id"] = data["verbal_id"].replace('grape vine', 'grapevine')
        data["verbal_id"] = data["verbal_id"].replace('air bag', 'airbag')
        data["verbal_id"] = data["verbal_id"].replace('jean', 'jeans')
        data["verbal_id"] = data["verbal_id"].replace('jeanss', 'jeans')
        data["verbal_id"] = data["verbal_id"].replace('boxing glove', 'boxing gloves')
        data["verbal_id"] = data["verbal_id"].replace('boxing glovess', 'boxing gloves')
        data["verbal_id"] = data["verbal_id"].replace('shell (projectile)', 'shell1')
        data["verbal_id"] = data["verbal_id"].replace('shell (animal housing)', 'shell2')
        data["verbal_id"] = data["verbal_id"].replace('shell (fruit/nut)', 'shell3')
        data["verbal_id"] = data["verbal_id"].replace('mold (form)', 'mold1')
        data["verbal_id"] = data["verbal_id"].replace('mold (fungus)', 'mold2')
        data["verbal_id"] = data["verbal_id"].replace('mouse (animal)', 'mouse1')
        data["verbal_id"] = data["verbal_id"].replace('mouse (computer)', 'mouse2')
        data["verbal_id"] = data["verbal_id"].replace('camera (photos)', 'camera1')
        data["verbal_id"] = data["verbal_id"].replace('camera (videos)', 'camera2')
        data["verbal_id"] = data["verbal_id"].replace('bat', 'bat2') # in this experiment there wasn't another bat
        data["verbal_id"] = data["verbal_id"].replace('bat (animal)', 'bat1')
        data["verbal_id"] = data["verbal_id"].replace('bat (sports)', 'bat2')
        data["verbal_id"] = data["verbal_id"].replace('baton (conductor', 'baton1')
        data["verbal_id"] = data["verbal_id"].replace('baton (police truncheon)', 'baton2')
        data["verbal_id"] = data["verbal_id"].replace('baton (relay running)', 'baton4')
        data["verbal_id"] = data["verbal_id"].replace('bow (decoration)', 'bow3')
        data["verbal_id"] = data["verbal_id"].replace('bow (weapon)', 'bow2')
        data["verbal_id"] = data["verbal_id"].replace('bow (hair knot)', 'bow1')
        data["verbal_id"] = data["verbal_id"].replace('bracelet (jewelry)', 'bracelet1')
        data["verbal_id"] = data["verbal_id"].replace('bracelet (wristband)', 'bracelet2')
        data["verbal_id"] = data["verbal_id"].replace('button (clothes button)', 'button1')
        data["verbal_id"] = data["verbal_id"].replace('button (device button)', 'button2')
        data["verbal_id"] = data["verbal_id"].replace('calf (animal)', 'calf1')
        data["verbal_id"] = data["verbal_id"].replace('calf (leg)', 'calf2')
        data["verbal_id"] = data["verbal_id"].replace('chest (body)', 'chest1')
        data["verbal_id"] = data["verbal_id"].replace('chest (box)', 'chest2')
        data["verbal_id"] = data["verbal_id"].replace('chicken (meat)', 'chicken1')
        data["verbal_id"] = data["verbal_id"].replace('chicken (animal)', 'chicken2')
        data["verbal_id"] = data["verbal_id"].replace('clipper (garden)', 'clipper1')
        data["verbal_id"] = data["verbal_id"].replace('clipper (nails)', 'clipper2')
        data["verbal_id"] = data["verbal_id"].replace('crystal (material)', 'crystal')
        data["verbal_id"] = data["verbal_id"].replace('crystal (rock)', 'crystal')
        data["verbal_id"] = data["verbal_id"].replace('hook (door hook)', 'hook1')
        data["verbal_id"] = data["verbal_id"].replace('hook (attachment hook)', 'hook2')
        data["verbal_id"] = data["verbal_id"].replace('juicer (juice squeezer)', 'juicer1')
        data["verbal_id"] = data["verbal_id"].replace('juicer (machine)', 'juicer2')
        data["verbal_id"] = data["verbal_id"].replace('file (card index)', 'file1')
        data["verbal_id"] = data["verbal_id"].replace('file (tool)', 'file2')
        data["verbal_id"] = data["verbal_id"].replace('pepper (seasoning)', 'pepper1')
        data["verbal_id"] = data["verbal_id"].replace('pepper (vegetable)', 'pepper2')
        data["verbal_id"] = data["verbal_id"].replace('pipe (smoking)', 'pipe1')
        data["verbal_id"] = data["verbal_id"].replace('pipe (tube)', 'pipe2')
        data["verbal_id"] = data["verbal_id"].replace('punch (beverage)', 'punch1')
        data["verbal_id"] = data["verbal_id"].replace('punch (hole punch)', 'punch2')
        data["verbal_id"] = data["verbal_id"].replace('screen (projection)', 'screen1')
        data["verbal_id"] = data["verbal_id"].replace('screen (net)', 'screen2')
        data["verbal_id"] = data["verbal_id"].replace('rack (shelf)', 'rack1')
        data["verbal_id"] = data["verbal_id"].replace('rack (stand)', 'rack2')
        data["verbal_id"] = data["verbal_id"].replace('stamp (postage stamp)', 'stamp1')
        data["verbal_id"] = data["verbal_id"].replace('stamp (pattern stamp)', 'stamp2')
        data["verbal_id"] = data["verbal_id"].replace('stove (food oven)', 'stove1')
        data["verbal_id"] = data["verbal_id"].replace('stove (heating oven)', 'stove2')
        data["verbal_id"] = data["verbal_id"].replace('straw (grain)', 'straw1')
        data["verbal_id"] = data["verbal_id"].replace('straw (drinking straw)', 'straw2')
        data["verbal_id"] = data["verbal_id"].replace('tank (storage)', 'tank1')
        data["verbal_id"] = data["verbal_id"].replace('tank (vehicle)', 'tank2')
        data["verbal_id"] = data["verbal_id"].replace('walker (for older adults)', 'walker1')
        data["verbal_id"] = data["verbal_id"].replace('walker (for toddlers)', 'walker2')
        data["verbal_id"] = data["verbal_id"].replace('chewing gum', 'gum')
        
        return data
    data = fix_polysemia(data)
    data = data.reset_index()
    
        # check if "merged_all_1002p.csv" file in +data contains the column "semantic_distance"
    semantic_distances = []
    # create empty column in "data" collad "semantic_distance
    data["semantic_distance"] = np.nan
    data["verbal_acc_corrected"] = np.nan

    print(len(data))
    for i in range(len(data)): # for each row in data
        # find index of image in uniqueID
        image_id = data["image_id"][i]
        index_true = uniqueID.index(data["image_id"][i])
        
        # find index of actual verbal response in uniqueID
        # if response is NaN, set distance to 
        
        if pd.isnull(data["verbal_id"][i]) or data["verbal_id"][i] == '':
            semantic_distances.append(np.nan)
            # set semantic distance in the data pd to NaN
            data.loc[i, "semantic_distance"] = np.nan
            if data["block_type"][i] == "recognition":
                data.loc[i, "verbal_acc_corrected"] = 0
            
            continue
        
        else: # check if verbal_id is in any row of synonyms
            # Loop through the list of synonyms and look for the image label
            data['verbal_id'] = data['verbal_id'].str.replace(' ', '_')
            word = data["verbal_id"][i].strip()
            pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
            
            found = False  # track whether we found a match

            
            for idx, synonym_string in enumerate(synonyms_distance):
                # Step 1: split and strip each synonym in the string
                candidates = [w.strip() for w in synonym_string.split(",")]

                # Step 2: check for full-word match
                matches = [s for s in candidates if pattern.fullmatch(s)]
                
                if matches:
                    index_response = idx
                    semantic_distances.append(dist_matrix[index_true, index_response])
                    data.loc[i, "semantic_distance"] = dist_matrix[index_true, index_response]
                    found = True
                    correct_id = [s for s in candidates if pattern.fullmatch(image_id)]
                    if correct_id:
                        data.loc[i, "verbal_acc_corrected"] = 1
                    else:
                        data.loc[i, "verbal_acc_corrected"] = 0
                    #print (f"✅ Found match for verbal_id '{word}' in synonyms list: {synonym_string}")
                    break

            if not found:
                print(f"⚠️ No match for subject {data['subject'][i]} with verbal_id '{word}'")
                raise ValueError("verbal_id not found in any synonym list as a full word")
                    
    
    return data
## main function
def main():
    parser = argparse.ArgumentParser(description="Preprocess a multi-subject JATOS .txt file.")
    parser.add_argument("--file", type=str, required=True, help="Filename of the raw .txt file (inside data/raw/)")
    args = parser.parse_args()

    file_path = RAW_DIR / args.file
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        return

    logging.info(f"📄 Processing file: {args.file}")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = split_into_chunks(lines)
    logging.info(f"✂️ Split into {len(chunks)} chunks")

    skipped_chunks = []
    success_count = 0
    
    # Track incomplete subjects
    incomplete_subjects = []
    incomplete_count = 0

    for i, chunk in enumerate(chunks):
        df, reason = process_chunk(chunk)
        if df is None:
            skipped_chunks.append((i + 1, reason))
            continue
        # preprocess verbal accuracy and semantic distance
        df = compute_sd(df)
        
        # save dataframe to csv
        subj = df['subject'].iloc[0]
        out_path = DERIV_DIR / f"subj_{subj}.csv"
        df.to_csv(out_path, index=False)
        logging.info(f"✅ Saved subj_{subj}.csv with {len(df)} rows")
        success_count += 1

        
        if len(df) < 195:
            incomplete_subjects.append((subj, len(df)))
            incomplete_count += 1

    

    log_path = DERIV_DIR / "preprocessing_log.txt"
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("Preprocessing log — flare-dot-task\n")
        logf.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        logf.write(f"Input file: {args.file}\n")
        logf.write(f"Total chunks found: {len(chunks)}\n")
        logf.write(f"Successful participant CSVs saved: {success_count}\n")
        if incomplete_subjects:
            logf.write(f"Subjects with incomplete data: {incomplete_count}\n")
        for subj_id, trial_count in incomplete_subjects:
            logf.write(f" - {subj_id} — {trial_count} trials\n")
            
        logf.write(f"\nSkipped: {len(skipped_chunks)}\n\n")

        if skipped_chunks:
            logf.write("Skipped chunks:\n")
            for chunk_num, reason in skipped_chunks:
                logf.write(f" - Chunk {chunk_num} — {reason}\n")
                
        logf.write(f"\n\nFINAL N OF GOOD PARTICIPANTS: {success_count - incomplete_count}\n")
        

if __name__ == "__main__":
    main()
